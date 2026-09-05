import torch
import pandas as pd
from collections import defaultdict
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from dataset_pipeline import AHCWindowDataset, DataLoader, sample_video_frames

BASE_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LORA_PATH = "ahc_vlm_lora_model_balanced"
VAL_CSV = "val_windows.csv"

def normalize_prediction(raw_text, valid_classes):
    raw_lower = raw_text.lower().strip()
    for cls in valid_classes:
        if cls in raw_lower:
            return cls
    return "unknown"

def evaluate_lora(num_samples=100):
    print("=" * 60)
    print("EVALUATING FINE-TUNED LoRA CHECKPOINT")
    print("=" * 60)
    
    print(f"Loading validation dataset from {VAL_CSV}...")
    dataset = AHCWindowDataset(VAL_CSV)
    # Seed generator for reproducible comparison against zero-shot
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator, num_workers=0)

    print(f"Loading base model: {BASE_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA weights from: {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    classes_str = ", ".join(dataset.classes)
    class_stats = defaultdict(lambda: {"total": 0, "correct": 0, "predicted_as": defaultdict(int)})
    total_correct = 0
    total_samples = 0

    print(f"\n--- Running Evaluation on {num_samples} Validation Samples ---\n")

    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        video_path = batch["video_path"][0]
        true_label = batch["label"][0]
        start_sec = batch["start_sec"].item()
        end_sec = batch["end_sec"].item()

        frames = sample_video_frames(video_path, start_sec, end_sec, num_frames=8)
        window_duration = end_sec - start_sec
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
        class_stats[true_label]["predicted_as"][normalized_pred] += 1
        total_samples += 1

        print(f"[{i+1}/{num_samples}] True: {true_label.ljust(30)} | Pred: {normalized_pred.ljust(30)} | {'✅' if is_match else '❌'}")

    # --- Summary Report ---
    print("\n" + "=" * 60)
    print("LoRA CHECKPOINT EVALUATION RESULTS")
    print("=" * 60)
    acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    print(f"Overall Accuracy: {acc:.1f}% ({total_correct}/{total_samples}) [Baseline was: 44.0%]\n")

    print("Per-Class Performance:")
    for cls in dataset.classes:
        stats = class_stats[cls]
        if stats["total"] > 0:
            c_acc = (stats["correct"] / stats["total"]) * 100
            print(f"  {cls.ljust(35)}: {c_acc:5.1f}% ({stats['correct']}/{stats['total']})")
            incorrect = {k: v for k, v in stats["predicted_as"].items() if k != cls and v > 0}
            if incorrect:
                mis_str = ", ".join([f"{k}: {v}" for k, v in incorrect.items()])
                print(f"    ↳ Confused with: {mis_str}")

if __name__ == "__main__":
    evaluate_lora(num_samples=100)