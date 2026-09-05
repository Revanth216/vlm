import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    master_csv = "training_windows.csv"
    try:
        df = pd.read_csv(master_csv)
    except FileNotFoundError:
        print(f"Error: Could not find {master_csv}")
        return

    print(f"Loading {len(df)} master windows...")

    # --- SAFETY CHECK 1: Multi-label videos ---
    video_label_counts = df.groupby('video_id')['label'].nunique()
    multi_label_vids = video_label_counts[video_label_counts > 1]
    
    if not multi_label_vids.empty:
        print(f"[WARNING] Found {len(multi_label_vids)} video(s) with multiple unique labels.")
        print("Resolving primary label by prioritizing anomalies over 'normal'...")
    else:
        print("[✓] All videos have a single unique label.")

    # Function to pick a safe primary label for stratification
    def get_primary_label(group):
        labels = group['label'].unique()
        if len(labels) == 1:
            return labels[0]
        # If both normal and anomalies exist in the same video, prioritize the anomaly
        anomalies = [l for l in labels if l != 'normal']
        return anomalies[0] if anomalies else labels[0]

    video_labels = df.groupby('video_id').apply(get_primary_label, include_groups=False).reset_index(name='label')
    
    print(f"Found {len(video_labels)} unique videos for partitioning.")

    # 2. Perform a Stratified 80/20 Split at the VIDEO level
    train_vids, val_vids = train_test_split(
        video_labels['video_id'], 
        test_size=0.20, 
        random_state=42,       # Locks seed for reproducibility
        stratify=video_labels['label']
    )

    # --- SAFETY CHECK 2: Zero Overlap Assertion ---
    train_set = set(train_vids)
    val_set = set(val_vids)
    overlap = train_set.intersection(val_set)
    
    print(f"  [{'✓' if len(overlap) == 0 else '✗'}] Train/Val video overlap count: {len(overlap)}")
    if len(overlap) > 0:
        raise ValueError("Data leakage detected! Training and validation sets share physical videos.")

    # 3. Map split videos back to windows
    train_df = df[df['video_id'].isin(train_vids)].copy()
    val_df = df[df['video_id'].isin(val_vids)].copy()

    # 4. Save outputs
    train_df.to_csv("train_windows.csv", index=False)
    val_df.to_csv("val_windows.csv", index=False)

    print("\n" + "="*50)
    print("DATASET SPLIT SUCCESSFUL & VERIFIED")
    print("="*50)
    print(f"TRAIN : {len(train_df)} windows (from {len(train_vids)} videos)")
    print(f"VAL   : {len(val_df)} windows (from {len(val_vids)} videos)")

    print("\n--- Validation Set Class Distribution ---")
    val_dist = val_df['label'].value_counts()
    for cls, count in val_dist.items():
        print(f"{cls.ljust(35)}: {count}")

if __name__ == "__main__":
    split_dataset()