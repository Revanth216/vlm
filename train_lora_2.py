import torch
import pandas as pd
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from unsloth import FastVisionModel
from unsloth.trainer import UnslothVisionDataCollator
from transformers import TrainingArguments, Trainer
from dataset_pipeline import sample_video_frames

# ==============================================================================
# 🚨 TRITON KERNEL BYPASS FOR PASCAL GPUs (Quadro P6000)
# Overwrite Unsloth's custom loss kernels with native PyTorch cross-entropy
# to prevent "device kernel image is invalid" errors.
# ==============================================================================
import unsloth.kernels.cross_entropy_loss
import unsloth_zoo.loss_utils

def fallback_fast_ce(logits, labels, **kwargs):
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

def fallback_unsloth_fixed_ce(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
    return F.cross_entropy(
        source.view(-1, source.size(-1)), 
        target.view(-1), 
        ignore_index=ignore_index
    )

unsloth.kernels.cross_entropy_loss.fast_cross_entropy_loss = fallback_fast_ce
unsloth_zoo.loss_utils.unsloth_fixed_cross_entropy = fallback_unsloth_fixed_ce
# ==============================================================================

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"

TRAIN_CSV = "train_windows.csv"
VAL_CSV = "val_windows.csv"

OUTPUT_DIR = "outputs"
FINAL_DIR = "ahc_vlm_lora_model_balanced"

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
print("LOADING & BALANCING DATASETS")
print("=" * 60)

train_df_raw = pd.read_csv(TRAIN_CSV)
print("Original Training Distribution:")
print(train_df_raw['label'].value_counts())

# Balance classes to the median count to prevent majority class dominance
TARGET_SAMPLES = int(train_df_raw['label'].value_counts().median())
print(f"\nBalancing all classes to exactly {TARGET_SAMPLES} samples...")

balanced_df = train_df_raw.groupby('label').sample(
    n=TARGET_SAMPLES, 
    replace=True, 
    random_state=42
)
# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced Training Distribution:")
print(balanced_df['label'].value_counts())

val_df = pd.read_csv(VAL_CSV)

# Updated PyTorch Dataset accepting a DataFrame directly
class AHCVideoDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.classes_str = ", ".join(sorted(self.df["label"].unique()))
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Extract 8 frames dynamically
        frames = sample_video_frames(
            row['video_path'], 
            float(row['window_start']), 
            float(row['window_end']), 
            num_frames=8
        )
        
        # Convert NumPy arrays to PIL Images for Unsloth's custom collator
        frames = [Image.fromarray(frame) for frame in frames]
        
        # Format the conversation with fps included to prevent warnings
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": 2.0},
                    {"type": "text", "text": f"You are an anomaly detection system. Classify this 4-second video window into exactly one of these classes: [{self.classes_str}]. Output strictly the class name."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": row['label']}
                ]
            }
        ]
        return {"messages": conversation}

train_dataset = AHCVideoDataset(balanced_df)
val_dataset = AHCVideoDataset(val_df)

print("\nTrain examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))

print("\n" + "=" * 60)
print("CREATING COLLATOR")
print("=" * 60)
data_collator = UnslothVisionDataCollator(model, tokenizer)

print("\n" + "=" * 60)
print("TRAINING CONFIGURATION")
print("=" * 60)

use_bf16 = torch.cuda.is_bf16_supported()
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_steps=35,               # Increased steps for the balanced dataset
    learning_rate=2e-4,
    warmup_steps=10,             # Increased warmup slightly for stability
    fp16=not use_bf16,
    bf16=use_bf16,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    logging_steps=5,             # Log less frequently now that steps are higher
    eval_strategy="no",          # Keep evaluation disabled for this run
    save_strategy="no",          # Save only at the end
    seed=42,
    report_to="none",
    remove_unused_columns=False, 
)

print("\n" + "=" * 60)
print("CREATING TRAINER (Standard Hugging Face Trainer)")
print("=" * 60)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=data_collator,
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