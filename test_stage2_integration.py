import cv2
import time
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from dataset_pipeline import sample_video_frames

# Configuration
TEST_VIDEO = "Train and Test/test/videos/T021.mp4"
BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LORA_PATH = "ahc_vlm_lora_model"

VALID_CLASSES = [
    "normal", "traffic_accident", "traffic_congestion", 
    "stalled_or_broken_down_vehicle", "vehicle_blocking_traffic", 
    "wrong_way_driving", "road_spill_or_debris", "waterlogging_or_flood", 
    "fire", "smoke", "fighting_or_violence", "loitering_or_suspicious_presence"
]
CLASSES_STR = ", ".join(sorted(VALID_CLASSES))

def detect_candidate_windows(video_path, motion_threshold_pct=1.0, window_size_sec=4.0, stride_sec=2.0, fps_target=2):
    """Stage 1: Fast motion gatekeeper returning candidate windows."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return []

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 25.0 
        
    frame_skip = int(orig_fps / fps_target)
    if frame_skip == 0:
        frame_skip = 1
        
    prev_gray = None
    frame_count = 0
    candidate_windows = []
    motion_buffer = []
    
    frames_per_window = int(window_size_sec * fps_target)
    frames_per_stride = int(stride_sec * fps_target)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
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
                        candidate_windows.append((window_start, window_end))
                        
                    motion_buffer = motion_buffer[frames_per_stride:]
            
            prev_gray = gray
        frame_count += 1

    cap.release()
    return candidate_windows

def normalize_prediction(raw_text):
    raw_lower = raw_text.lower().strip()
    for cls in VALID_CLASSES:
        if cls in raw_lower:
            return cls
    return raw_lower

def run_integration_test():
    print("=" * 60)
    print("RUNNING STAGE 1 + STAGE 2 VLM INTEGRATION TEST")
    print("=" * 60)
    
    # 1. Run Stage 1 to get candidates
    print(f"Scanning {TEST_VIDEO} with Stage 1...")
    candidates = detect_candidate_windows(TEST_VIDEO)
    print(f"Stage 1 identified {len(candidates)} candidate windows.\n")
    
    if not candidates:
        print("No candidates found. Try lowering motion_threshold_pct.")
        return

    # 2. Load Base Model + LoRA Adapter
    print(f"Loading processor and base model ({BASE_MODEL_ID})...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA weights from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    # 3. Process each candidate window with adaptive 4-8 FPS extraction
    print("\n" + "=" * 60)
    print("PROCESSING CANDIDATES THROUGH LoRA VLM")
    print("=" * 60)

    for idx, (start_sec, end_sec) in enumerate(candidates):
        window_duration = end_sec - start_sec
        # Adaptive extraction: 4 FPS -> 16 frames for a 4-second window
        target_fps = 4.0
        num_frames = int(window_duration * target_fps)
        
        # Extract frames as numpy arrays and convert to PIL
        raw_frames = sample_video_frames(TEST_VIDEO, start_sec, end_sec, num_frames=num_frames)
        pil_frames = [Image.fromarray(f) for f in raw_frames]
        seq_fps = len(pil_frames) / window_duration if window_duration > 0 else target_fps

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

        start_inf = time.time()
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=15)
        inf_time = (time.time() - start_inf) * 1000

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_pred = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        final_pred = normalize_prediction(raw_pred)

        print(f"Window {idx+1}: [{start_sec:.1f}s -> {end_sec:.1f}s] | Frames: {len(pil_frames)} | Pred: {final_pred.ljust(35)} | Time: {inf_time:.1f}ms")

if __name__ == "__main__":
    run_integration_test()