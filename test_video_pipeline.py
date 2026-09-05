import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import cv2
import numpy as np

class AHCWindowDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.classes = sorted(self.df["label"].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Safely parse the boolean string from the CSV
        needs_padding = str(row.get("needs_padding", False)).strip().lower() == "true"
        
        return {
            "video_path": row["video_path"],
            "start_sec": float(row["window_start"]),
            "end_sec": float(row["window_end"]),
            "label": row["label"],
            "label_id": self.class_to_idx[row["label"]],
            "needs_padding": needs_padding,
        }

def create_balanced_sampler(dataset):
    labels = dataset.df["label"]
    class_counts = labels.value_counts()
    
    sample_weights = labels.map(lambda label: 1.0 / class_counts[label])
    sample_weights = torch.as_tensor(sample_weights.to_numpy(), dtype=torch.double)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def sample_video_frames(video_path, start_sec, end_sec, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps

    actual_end_sec = min(end_sec, video_duration)
    actual_start_sec = min(start_sec, actual_end_sec)

    if actual_end_sec <= actual_start_sec:
        timestamps = [actual_start_sec] * num_frames
    else:
        step = (actual_end_sec - actual_start_sec) / num_frames
        timestamps = [actual_start_sec + (i + 0.5) * step for i in range(num_frames)]

    frames = []
    for t in timestamps:
        frame_idx = int(t * fps)
        frame_idx = min(frame_idx, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            if len(frames) == 0:
                cap.release()
                raise RuntimeError(f"Failed to read the very first frame from {video_path} at {t}s. Check video integrity.")
            else:
                # Duplicate the previous valid frame
                frames.append(frames[-1])

    cap.release()
    
    # Final safety check in case the loop completes but is short
    while len(frames) < num_frames:
        if len(frames) == 0:
            raise RuntimeError(f"Could not extract any frames from {video_path}.")
        frames.append(frames[-1])

    return frames

if __name__ == "__main__":
    # 1. Load training dataset
    train_dataset = AHCWindowDataset("train_windows.csv")

    print("Dataset loaded successfully")
    print("Total windows:", len(train_dataset))
    print("Classes:", train_dataset.classes)

    # 2. Test one ordinary sample
    sample = train_dataset[0]

    print("\nSample metadata:")
    print("Video:", sample["video_path"])
    print("Start:", sample["start_sec"])
    print("End:", sample["end_sec"])
    print("Label:", sample["label"])
    print("Label ID:", sample["label_id"])
    print("Needs padding:", sample["needs_padding"])

    # 3. Extract 8 frames
    frames = sample_video_frames(
        sample["video_path"],
        sample["start_sec"],
        sample["end_sec"],
        num_frames=8
    )

    print("\nFrame extraction:")
    print("Number of frames:", len(frames))

    for i, frame in enumerate(frames):
        print(
            f"Frame {i}: "
            f"shape={frame.shape}, "
            f"dtype={frame.dtype}, "
            f"min={frame.min()}, "
            f"max={frame.max()}"
        )

    # 4. Test balanced sampler
    sampler = create_balanced_sampler(train_dataset)

    print("\nBalanced sampler:")
    print("Sampler samples per epoch:", sampler.num_samples)

    # 5. Test DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=0
    )

    batch = next(iter(train_loader))

    print("\nDataLoader test successful")
    print("Batch labels:", batch["label"])
    print("Batch label IDs:", batch["label_id"])