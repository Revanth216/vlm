import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments
from trl import SFTTrainer
from dataset_pipeline import sample_video_frames

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

TRAIN_CSV = "train_windows.csv"
VAL_CSV = "val_windows.csv"

OUTPUT_DIR = "outputs"
FINAL_DIR = "ahc_vlm_lora_model"

print("=" * 60)
print("GPU CHECK")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GB:", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
))

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_NAME,
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)

print("Model loaded successfully.")

print("\n" + "=" * 60)
print("ADDING LORA")
print("=" * 60)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)

print("LoRA configured successfully.")

print("\n" + "=" * 60)
print("LOADING DATASETS (Hugging Face Format)")
print("=" * 60)

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
classes_str = ", ".join(sorted(train_df["label"].unique()))

# Load pandas DataFrames into Hugging Face Datasets
train_hf = Dataset.from_pandas(train_df)
val_hf = Dataset.from_pandas(val_df)

def format_vision_dataset(batch):
    """
    Transforms CSV metadata into Qwen-VL conversations on-the-fly.
    Extracts frames only when a batch is requested to save memory.
    """
    batch_messages = []
    for i in range(len(batch['video_path'])):
        video_path = batch['video_path'][i]
        start_sec = float(batch['window_start'][i])
        end_sec = float(batch['window_end'][i])
        label = batch['label'][i]
        
        # Extract 8 frames using your verified pipeline
        frames = sample_video_frames(video_path, start_sec, end_sec, num_frames=8)
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": f"You are an anomaly detection system. Classify this 4-second video window into exactly one of these classes: [{classes_str}]. Output strictly the class name."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": label}
                ]
            }
        ]
        batch_messages.append(conversation)
        
    return {"messages": batch_messages}

# Attach the lazy transform
train_hf.set_transform(format_vision_dataset)
val_hf.set_transform(format_vision_dataset)

print("Train examples:", len(train_hf))
print("Validation examples:", len(val_hf))

print("\n" + "=" * 60)
print("CREATING COLLATOR")
print("=" * 60)
# Unsloth's Vision Collator will handle the tokenization and image processing
data_collator = UnslothVisionDataCollator(model, tokenizer)

print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

use_bf16 = torch.cuda.is_bf16_supported()
print("BF16 supported:", use_bf16)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=20, # Short smoke test
    learning_rate=2e-4,
    warmup_steps=2,
    fp16=not use_bf16,
    bf16=use_bf16,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    logging_steps=1,
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,
    seed=42,
    report_to="none",
    remove_unused_columns=False, # Critical for vision training
)

print("\n" + "=" * 60)
print("CREATING TRAINER")
print("=" * 60)

# Dummy function to satisfy Unsloth's strict validation check
def dummy_formatting(example):
    return [""]

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_hf,
    eval_dataset=val_hf,
    data_collator=data_collator,
    
    # Pass the dummy function and force the trainer to skip its default processing
    formatting_func=dummy_formatting,
    dataset_kwargs={"skip_prepare_dataset": True},
    
    args=training_args,
)

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

trainer.train()

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("Saved to:", FINAL_DIR)
print("TRAINING COMPLETE")