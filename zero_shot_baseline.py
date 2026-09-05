import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from dataset_pipeline import AHCWindowDataset, DataLoader, sample_video_frames
from collections import defaultdict

def normalize_prediction(raw_text, valid_classes):
    """Fuzzy matching to extract the correct class if the VLM is overly chatty."""
    raw_lower = raw_text.lower()
    for cls in valid_classes:
        if cls in raw_lower:
            return cls
    return "unknown"

def run_zero_shot_baseline(csv_path="val_windows.csv", model_id="Qwen/Qwen3-VL-4B-Instruct", num_samples=5):
    print(f"Loading validation dataset from {csv_path}...")
    dataset = AHCWindowDataset(csv_path)
    # Shuffle=True ensures our 5 samples are a random cross-section
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    print(f"Loading processor and model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Swapped to float16 to support older Pascal architecture GPUs like the Quadro P6000
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()

    classes_str = ", ".join(dataset.classes)
    
    class_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    total_correct = 0
    total_samples = 0
    
    print(f"\n--- Starting Stage A: Smoke Test ({num_samples} samples) ---\n")
    
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break
            
        video_path = batch["video_path"][0]
        true_label = batch["label"][0]
        start_sec = batch["start_sec"].item()
        end_sec = batch["end_sec"].item()
        
        frames = sample_video_frames(video_path, start_sec, end_sec, num_frames=8)
        
        # Dynamically calculate the actual FPS of this specific frame sequence
        window_duration = end_sec - start_sec
        # Fallback to 2.0 to prevent division by zero on corrupted timestamps
        seq_fps = len(frames) / window_duration if window_duration > 0.0 else 2.0
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": seq_fps}, 
                    {"type": "text", "text": f"You are an anomaly detection system. Classify this 4-second video window into exactly one of these classes: [{classes_str}]. Output strictly the class name and nothing else."}
                ]
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Explicitly pass the dynamic sequence metadata to bypass the warning and prevent resampling
        # Explicitly pass BOTH fps and total_num_frames to satisfy VideoMetadata
        inputs = processor(
            text=[text], 
            videos=[frames], 
            video_metadata=[{"fps": seq_fps, "total_num_frames": len(frames)}],
            padding=True, 
            cap_pixels_per_frame=True, 
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        raw_prediction = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        normalized_pred = normalize_prediction(raw_prediction, dataset.classes)
        
        is_match = (normalized_pred == true_label)
        if is_match: 
            total_correct += 1
            class_stats[true_label]["correct"] += 1
        class_stats[true_label]["total"] += 1
        total_samples += 1
            
        print(f"Sample {i+1}:")
        print(f"  True Label : {true_label}")
        print(f"  Raw Output : {raw_prediction}")
        print(f"  Normalized : {normalized_pred}")
        print(f"  Match      : {'✅' if is_match else '❌'}\n")
        
    print("\n--- Baseline Results ---")
    print(f"Overall Accuracy: {(total_correct/total_samples)*100:.1f}% ({total_correct}/{total_samples})")
    
    print("\nPer-Class Breakdown:")
    for cls in dataset.classes:
        stats = class_stats[cls]
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"  {cls.ljust(35)}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    run_zero_shot_baseline(num_samples=100)