import os
import json
import time
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from dataset_pipeline import sample_video_frames

# ==========================================
# CONFIGURATION
# ==========================================
TEST_VIDEO = "Train and Test/test/videos/T021.mp4"
VIDEO_ID = "T021"
VIDEO_LEVEL = 2  # Set evaluation level (1, 2, or 3)

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
    """Samples video at 2 FPS and flags sliding windows with motion >= threshold."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 0, 0

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
        if not ret:
            break
            
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
    """
    Conservative temporal suppression:
    - Retains significant local motion peaks.
    - Suppresses redundant intermediate windows in uniform motion plateaus.
    - Guarantees no temporal blind spots (adjacent gap <= max_allowed_gap_sec).
    - Reduces 8 candidates down to ~4-6 windows.
    """
    if len(candidates) <= 2:
        return candidates

    windows = sorted(candidates, key=lambda x: x[0])
    selected = []
    i = 0

    while i < len(windows):
        selected.append(windows[i])

        if i + 1 >= len(windows):
            break

        next_w = windows[i + 1]

        # Evaluate if skipping next_w leaves continuous coverage
        if i + 2 < len(windows):
            next_next_w = windows[i + 2]
            # Gap between end of current selected and start of next-next
            gap_if_skipped = next_next_w[0] - windows[i][1]

            curr_motion = windows[i][2]
            next_motion = next_w[2]
            next_next_motion = next_next_w[2]

            # Keep next_w if it is a prominent motion spike
            is_peak = (next_motion > curr_motion * 1.25) and (next_motion > next_next_motion * 1.25)

            if is_peak or gap_if_skipped > max_allowed_gap_sec:
                i += 1  # Retain next window
            else:
                i += 2  # Safely step over intermediate plateau window
        else:
            # For the last pair, keep the tail window if it extends coverage
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
        if cls in raw_lower:
            return cls
    return "normal"


# ==========================================
# STAGE 3: TEMPORAL STITCHING & HYSTERESIS
# ==========================================
def temporal_stitch_with_hysteresis(window_results, video_duration):
    """
    Stitches consecutive detections of the same class.
    Allows a 1-window normal bridge if physically adjacent.
    Never merges different anomaly classes.
    """
    if not window_results:
        return []

    stitched_intervals = []
    active_class = None
    start_time = None
    end_time = None

    sorted_windows = sorted(window_results, key=lambda x: x['start'])

    for i in range(len(sorted_windows)):
        w = sorted_windows[i]

        if active_class is None:
            if w['pred'] != 'normal':
                active_class = w['pred']
                start_time = w['start']
                end_time = w['end']
        else:
            prev_w = sorted_windows[i - 1]
            time_jump = w['start'] - prev_w['start']

            # Large temporal jump breaks continuity
            if time_jump > 4.5:
                stitched_intervals.append({
                    "class_name": active_class,
                    "start_time_sec": round(start_time, 2),
                    "end_time_sec": round(min(end_time, video_duration), 2)
                })
                if w['pred'] != 'normal':
                    active_class = w['pred']
                    start_time = w['start']
                    end_time = w['end']
                else:
                    active_class = None
                continue

            if w['pred'] == active_class:
                end_time = max(end_time, w['end'])
            elif w['pred'] == 'normal':
                # Check 1-window normal bridge
                if i + 1 < len(sorted_windows):
                    next_w = sorted_windows[i + 1]
                    next_diff = next_w['start'] - w['start']
                    if next_w['pred'] == active_class and next_diff <= 2.5:
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
                # Class boundary change
                stitched_intervals.append({
                    "class_name": active_class,
                    "start_time_sec": round(start_time, 2),
                    "end_time_sec": round(min(end_time, video_duration), 2)
                })
                active_class = w['pred']
                start_time = w['start']
                end_time = w['end']

    if active_class is not None:
        stitched_intervals.append({
            "class_name": active_class,
            "start_time_sec": round(start_time, 2),
            "end_time_sec": round(min(end_time, video_duration), 2)
        })

    return stitched_intervals


# ==========================================
# LEVEL-AWARE OUTPUT FORMATTER
# ==========================================
def format_submission_output(video_id, level, stitched_events, runtime_meta):
    """
    Strict formatting rules:
    - Normal videos -> events: [] (never include class_name: normal).
    - Level 1 -> timestamps must be null.
    - Level 2/3 -> timestamps must be valid numbers (end > start).
    """
    formatted_events = []

    for ev in stitched_events:
        if ev['class_name'] == 'normal':
            continue

        if level == 1:
            formatted_events.append({
                "class_name": ev['class_name'],
                "start_time_sec": None,
                "end_time_sec": None
            })
        else:
            formatted_events.append({
                "class_name": ev['class_name'],
                "start_time_sec": ev['start_time_sec'],
                "end_time_sec": ev['end_time_sec']
            })

    output_record = {
        "video_id": video_id,
        "level": level,
        "events": formatted_events,
        "runtime_metadata": runtime_meta
    }
    return output_record


# ==========================================
# EXECUTION PIPELINE
# ==========================================
def run_final_pipeline():
    total_start_time = time.time()

    print("=" * 65)
    print(f"RUNNING FINAL PIPELINE TEST ON: {VIDEO_ID} (Level {VIDEO_LEVEL})")
    print("=" * 65)

    # 1. Stage 1: Motion Scan
    raw_candidates, frames_scanned, vid_dur = detect_candidate_windows(TEST_VIDEO)
    print(f"Stage 1 Raw Scan: {len(raw_candidates)} windows flagged (Frames scanned: {frames_scanned})")

    # 2. Candidate Reduction
    candidates = conservative_temporal_nms(raw_candidates)
    print(f"After Conservative NMS: {len(candidates)} candidate windows retained for VLM verification\n")

    if not candidates:
        print("No candidates survived gatekeeper. Classifying as normal.")
        runtime_meta = {
            "frames_processed": frames_scanned,
            "chunks_processed": 0,
            "end_to_end_internal_time_ms": round((time.time() - total_start_time) * 1000, 2),
            "model_runtimes": {"stage1_ms": round((time.time() - total_start_time) * 1000, 2)}
        }
        res = format_submission_output(VIDEO_ID, VIDEO_LEVEL, [], runtime_meta)
        print(json.dumps(res, indent=2))
        return

    # 3. Model Loading
    print(f"Loading {BASE_MODEL_ID} + LoRA ({LORA_PATH})...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    window_results = []
    vlm_durations = []
    total_vlm_frames = 0

    print("\n" + "=" * 65)
    print("STAGE 2: VLM VERIFICATION (ADAPTIVE 4/8 FPS)")
    print("=" * 65)

    for idx, (start_sec, end_sec, max_motion) in enumerate(candidates):
        w_dur = end_sec - start_sec
        target_fps = 8.0 if max_motion >= 5.0 else 4.0
        num_frames = int(w_dur * target_fps)

        raw_frames = sample_video_frames(TEST_VIDEO, start_sec, end_sec, num_frames=num_frames)
        pil_frames = [Image.fromarray(f) for f in raw_frames]
        total_vlm_frames += len(pil_frames)
        seq_fps = len(pil_frames) / w_dur if w_dur > 0 else target_fps

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": seq_fps},
                    {"type": "text", "text": f"You are an anomaly detection system. Classify this video window into exactly one of these classes: [{CLASSES_STR}]. Output strictly the class name and nothing else."}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            videos=[pil_frames],
            video_metadata=[{"fps": seq_fps, "total_num_frames": len(pil_frames)}],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        inf_start = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=15)
        inf_ms = (time.time() - inf_start) * 1000
        vlm_durations.append(inf_ms)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_pred = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        final_pred = normalize_prediction(raw_pred)

        print(f"Window {idx+1}/{len(candidates)}: [{start_sec:.1f}s -> {end_sec:.1f}s] | Motion: {max_motion:5.2f}% | FPS: {target_fps} ({len(pil_frames)} frames) | Pred: {final_pred.ljust(30)} | Time: {inf_ms:.1f}ms")

        window_results.append({
            "start": start_sec,
            "end": end_sec,
            "pred": final_pred,
            "motion": max_motion,
            "fps": target_fps,
            "frames": len(pil_frames),
            "inference_time": inf_ms
        })

    # 4. Stage 3: Temporal Stitching
    stitched_events = temporal_stitch_with_hysteresis(window_results, vid_dur)

    total_pipeline_time_ms = (time.time() - total_start_time) * 1000

    runtime_meta = {
        "frames_processed": frames_scanned + total_vlm_frames,
        "chunks_processed": len(candidates),
        "end_to_end_internal_time_ms": round(total_pipeline_time_ms, 2),
        "model_runtimes": {
            "vlm_mean_ms": round(sum(vlm_durations) / len(vlm_durations), 2) if vlm_durations else 0,
            "vlm_total_ms": round(sum(vlm_durations), 2)
        }
    }

    # 5. Format Output
    submission_record = format_submission_output(VIDEO_ID, VIDEO_LEVEL, stitched_events, runtime_meta)

    print("\n" + "=" * 65)
    print("FINAL LEVEL-AWARE SUBMISSION JSON:")
    print("=" * 65)
    print(json.dumps(submission_record, indent=2))


if __name__ == "__main__":
    run_final_pipeline()