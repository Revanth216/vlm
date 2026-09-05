import os
# --- MEMORY FRAGMENTATION FIX ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import json
import time
import cv2
import torch
import pandas as pd
from collections import Counter
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# Attempt to import frame sampler, otherwise define a fallback for safety
try:
    from dataset_pipeline import sample_video_frames
except ImportError:
    def sample_video_frames(video_path, start_sec, end_sec, num_frames=8):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
        total_frames_in_window = (end_sec - start_sec) * fps
        step = max(1, int(total_frames_in_window / num_frames))
        
        count = 0
        while len(frames) < num_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if count % step == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            count += 1
        cap.release()
        
        while len(frames) < num_frames and frames:
            frames.append(frames[-1].copy())
        return frames

# ==========================================
# CONFIGURATION
# ==========================================
TEST_DIR = os.path.join("Train and Test", "test")
VIDEOS_CSV = os.path.join(TEST_DIR, "videos.csv")
VIDEOS_FOLDER = os.path.join(TEST_DIR, "videos")
SUBMISSION_FILE = "predictions.json"

BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LORA_PATH = "ahc_vlm_lora_model"

VALID_CLASSES = [
    "normal", "traffic_accident", "traffic_congestion", 
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic", 
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood", 
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence"
]
CLASSES_STR = ", ".join(sorted(VALID_CLASSES))


# ==========================================
# STAGE 1: MOTION SCAN & CANDIDATE REDUCTION
# ==========================================
def detect_candidate_windows(video_path, motion_threshold_pct=1.0, window_size_sec=4.0, stride_sec=2.0, fps_target=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return [], 0, 0

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_dur = total_frames / orig_fps if orig_fps > 0 else 0
    
    frame_skip = max(1, int(orig_fps / fps_target))
    prev_gray = None
    frame_count = 0
    candidate_windows = []
    motion_buffer = []
    
    frames_per_window = int(window_size_sec * fps_target)
    frames_per_stride = int(stride_sec * fps_target)
    sampled_frames_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
            
        if frame_count % frame_skip == 0:
            sampled_frames_count += 1
            current_sec = frame_count / orig_fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_gray is not None:
                frame_delta = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(thresh)
                total_pixels = thresh.shape[0] * thresh.shape[1]
                motion_pct = (motion_pixels / total_pixels) * 100
                
                motion_buffer.append((current_sec, motion_pct))
                
                if len(motion_buffer) == frames_per_window:
                    window_start = motion_buffer[0][0]
                    window_end = motion_buffer[-1][0]
                    max_motion = max([m for _, m in motion_buffer])
                    
                    if max_motion >= motion_threshold_pct:
                        candidate_windows.append((window_start, window_end, max_motion))
                        
                    motion_buffer = motion_buffer[frames_per_stride:]
            
            prev_gray = gray
        frame_count += 1

    cap.release()
    return candidate_windows, sampled_frames_count, video_dur

def conservative_temporal_nms(candidates, max_allowed_gap_sec=1.0):
    if len(candidates) <= 2: return candidates
    windows = sorted(candidates, key=lambda x: x[0])
    selected = []
    i = 0

    while i < len(windows):
        selected.append(windows[i])
        if i + 1 >= len(windows): break

        next_w = windows[i + 1]
        if i + 2 < len(windows):
            next_next_w = windows[i + 2]
            gap_if_skipped = next_next_w[0] - windows[i][1]
            curr_motion, next_motion, next_next_motion = windows[i][2], next_w[2], next_next_w[2]

            is_peak = (next_motion > curr_motion * 1.25) and (next_motion > next_next_motion * 1.25)

            if is_peak or gap_if_skipped > max_allowed_gap_sec:
                i += 1
            else:
                i += 2
        else:
            if next_w[1] > windows[i][1]:
                selected.append(next_w)
            break
    return selected


# ==========================================
# STAGE 2: VLM INFERENCE HELPERS
# ==========================================
def normalize_prediction(raw_text):
    raw_lower = raw_text.lower().strip()
    for cls in VALID_CLASSES:
        if cls in raw_lower: return cls
    return "normal"


# ==========================================
# STAGE 3: LEVEL SPECIFIC LOGIC
# ==========================================
def aggregate_level_1(window_results):
    if not window_results:
        return "normal"
        
    preds = [w['pred'] for w in window_results]
    anomalies = [p for p in preds if p != 'normal']
    
    if anomalies:
        most_common = Counter(anomalies).most_common(1)[0][0]
        return most_common
    return "normal"

def temporal_stitch_with_hysteresis(window_results, video_duration):
    if not window_results: return []

    stitched_intervals = []
    active_class = None
    start_time, end_time = None, None
    sorted_windows = sorted(window_results, key=lambda x: x['start'])

    for i, w in enumerate(sorted_windows):
        if active_class is None:
            if w['pred'] != 'normal':
                active_class, start_time, end_time = w['pred'], w['start'], w['end']
        else:
            prev_w = sorted_windows[i - 1]
            time_jump = w['start'] - prev_w['start']

            if time_jump > 4.5:
                stitched_intervals.append({
                    "class_name": active_class,
                    "start_time_sec": round(start_time, 2),
                    "end_time_sec": round(min(end_time, video_duration), 2)
                })
                if w['pred'] != 'normal':
                    active_class, start_time, end_time = w['pred'], w['start'], w['end']
                else:
                    active_class = None
                continue

            if w['pred'] == active_class:
                end_time = max(end_time, w['end'])
            elif w['pred'] == 'normal':
                if i + 1 < len(sorted_windows):
                    next_w = sorted_windows[i + 1]
                    if next_w['pred'] == active_class and (next_w['start'] - w['start']) <= 2.5:
                        end_time = max(end_time, w['end'])
                    else:
                        stitched_intervals.append({
                            "class_name": active_class,
                            "start_time_sec": round(start_time, 2),
                            "end_time_sec": round(min(end_time, video_duration), 2)
                        })
                        active_class = None
                else:
                    stitched_intervals.append({
                        "class_name": active_class,
                        "start_time_sec": round(start_time, 2),
                        "end_time_sec": round(min(end_time, video_duration), 2)
                    })
                    active_class = None
            else:
                stitched_intervals.append({
                    "class_name": active_class,
                    "start_time_sec": round(start_time, 2),
                    "end_time_sec": round(min(end_time, video_duration), 2)
                })
                active_class, start_time, end_time = w['pred'], w['start'], w['end']

    if active_class is not None:
        stitched_intervals.append({
            "class_name": active_class,
            "start_time_sec": round(start_time, 2),
            "end_time_sec": round(min(end_time, video_duration), 2)
        })

    return stitched_intervals


# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_all():
    script_start_time = time.time()
    
    # 1. Read Test Metadata
    if not os.path.exists(VIDEOS_CSV):
        print(f"Error: {VIDEOS_CSV} not found.")
        return
        
    df_videos = pd.read_csv(VIDEOS_CSV)
    
    video_to_level = {}
    GT_CSV = os.path.join(TEST_DIR, "ground_truth.csv")
    
    if os.path.exists(GT_CSV):
        df_gt = pd.read_csv(GT_CSV)
        df_gt.columns = df_gt.columns.str.lower()
        if 'video_id' in df_gt.columns and 'level' in df_gt.columns:
            video_to_level = df_gt.drop_duplicates(subset=['video_id']).set_index('video_id')['level'].to_dict()
    
    # --- SUBMISSION-READY CHECKPOINT LOADING ---
    all_predictions = []
    processed_vids = set()
    
    if os.path.exists(SUBMISSION_FILE):
        print(f"Found existing submission file: {SUBMISSION_FILE}")
        try:
            with open(SUBMISSION_FILE, "r") as f:
                data = json.load(f)
                all_predictions = data.get("predictions", [])
            processed_vids = {p['video_id'] for p in all_predictions}
            print(f"Loaded {len(processed_vids)} completed videos. Resuming...\n")
        except Exception as e:
            print(f"Warning: Could not parse {SUBMISSION_FILE}: {e}")
            print("Starting fresh...")
    else:
        print("No prior submission found. Starting fresh from the beginning...")
    # -------------------------------------------

    # 2. Load Models
    print(f"Loading processor and model ({BASE_MODEL_ID}) + LoRA...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 3. Process Each Video
    for _, row in df_videos.iterrows():
        vid_id = row['video_id']
        
        if vid_id in processed_vids:
            print(f"Skipping {vid_id} (Already in predictions.json)")
            continue

        level = int(video_to_level.get(vid_id, 3))
        vid_path = os.path.join(VIDEOS_FOLDER, f"{vid_id}.mp4")
        
        if not os.path.exists(vid_path):
            print(f"Warning: {vid_path} missing. Skipping.")
            continue

        print("\n" + "="*50)
        print(f"Processing Video: {vid_id} (Level {level})")
        print("="*50)
        
        vid_start_time = time.time()
        
        # Stage 1 & Candidate Reduction
        raw_candidates, frames_scanned, vid_dur = detect_candidate_windows(vid_path)
        candidates = conservative_temporal_nms(raw_candidates)
        
        window_results = []
        vlm_durations = []
        total_vlm_frames = 0
        
        # Stage 2: VLM Inference
        for start_sec, end_sec, max_motion in candidates:
            w_dur = end_sec - start_sec
            
            # --- FIX: HARD 8-FRAME CAP & 512x512 RESIZE TO PREVENT OOM ---
            num_frames = 8
            raw_frames = sample_video_frames(vid_path, start_sec, end_sec, num_frames=num_frames)
            pil_frames = [Image.fromarray(f).resize((512, 512)) for f in raw_frames]
            
            total_vlm_frames += len(pil_frames)
            seq_fps = len(pil_frames) / w_dur if w_dur > 0 else 2.0
            # -------------------------------------------------------------

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": seq_fps},
                    {"type": "text", "text": f"You are an anomaly detection system. Classify this video window into exactly one of these classes: [{CLASSES_STR}]. Output strictly the class name and nothing else."}
                ]
            }]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], videos=[pil_frames],
                video_metadata=[{"fps": seq_fps, "total_num_frames": len(pil_frames)}],
                padding=True, return_tensors="pt"
            ).to(model.device)

            inf_start = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=15)
            inf_ms = (time.time() - inf_start) * 1000
            vlm_durations.append(inf_ms)

            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            final_pred = normalize_prediction(processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip())

            window_results.append({
                "start": start_sec, "end": end_sec, "pred": final_pred
            })
            
            # --- MEMORY CLEANUP ---
            del inputs, generated_ids, generated_ids_trimmed, raw_frames, pil_frames, messages, text
            gc.collect()
            torch.cuda.empty_cache()

        # Stage 3: Level-Aware Aggregation & Stitching
        final_events = []
        if level == 1:
            overall_class = aggregate_level_1(window_results)
            if overall_class != 'normal':
                final_events.append({
                    "class_name": overall_class,
                    "start_time_sec": None,
                    "end_time_sec": None
                })
        else:
            stitched = temporal_stitch_with_hysteresis(window_results, vid_dur)
            for ev in stitched:
                if ev['class_name'] != 'normal':
                    final_events.append({
                        "class_name": ev['class_name'],
                        "start_time_sec": ev['start_time_sec'],
                        "end_time_sec": ev['end_time_sec']
                    })

        vid_total_ms = (time.time() - vid_start_time) * 1000
        
        # --- FIX: OFFICIAL MODEL RUNTIMES FORMAT ---
        call_count = len(vlm_durations)
        total_vlm_ms = round(sum(vlm_durations), 2)
        avg_vlm_ms = round(total_vlm_ms / call_count, 2) if call_count > 0 else 0
        
        runtime_meta = {
            "frames_processed": frames_scanned + total_vlm_frames,
            "chunks_processed": len(candidates),
            "end_to_end_internal_time_ms": round(vid_total_ms, 2),
            "model_runtimes": [
                {
                    "model_name": "Qwen3-VL-4B-Instruct-LoRA",
                    "call_count": call_count,
                    "total_time_ms": total_vlm_ms,
                    "average_time_ms": avg_vlm_ms
                }
            ] if call_count > 0 else []
        }
        # -------------------------------------------

        all_predictions.append({
            "video_id": vid_id,
            "events": final_events,
            "runtime_metadata": runtime_meta
        })

        # --- INSTANT CHECKPOINT SAVE (Full Submission Schema) ---
        final_submission = {
            "schema_version": "1.0",
            "submission_id": "ahc-qwen3-lora-run-01",
            "model_name": "Qwen3-VL-4B-Instruct-LoRA",
            "run_metadata": {
                "description": "End-to-end evaluation using 8-frame VLM sampling and temporal hysteresis stitching"
            },
            "total_wall_time_ms": round((time.time() - script_start_time) * 1000, 2),
            "hardware": "1x NVIDIA P6000 (24GB)",
            "predictions": all_predictions
        }
        
        with open(SUBMISSION_FILE, "w") as f:
            json.dump(final_submission, f, indent=2)
            
        print(f"Video {vid_id} complete. Valid submission saved to {SUBMISSION_FILE}.")
        
        # Debugging Summary
        print(f"Raw candidates: {len(raw_candidates)}")
        print(f"VLM candidates: {len(candidates)}")
        print("VLM predictions:")
        for w in window_results:
            print(f"    {w['pred']}")
        if level == 1:
            print("Final Level-1 prediction:")
            print(f"    {overall_class if 'overall_class' in locals() else 'normal'}")
        else:
            print(f"Final Level-{level} stitched events:")
            for ev in final_events:
                print(f"    {ev['class_name']} [{ev['start_time_sec']}s -> {ev['end_time_sec']}s]")
        print("Runtime:")
        print(f"    Total: {vid_total_ms:.1f}ms | VLM Total: {total_vlm_ms:.1f}ms")
        
        # --- VIDEO LEVEL MEMORY CLEANUP ---
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "="*50)
    print(f"Successfully processed {len(all_predictions)} videos.")
    print(f"Final predictions saved to {SUBMISSION_FILE}")
    print("="*50)

if __name__ == "__main__":
    run_all()