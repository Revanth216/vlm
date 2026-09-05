import os
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict

def get_video_duration(video_path):
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fc / fps if fps > 0 else 0

def summarize_violations():
    train_dir = os.path.join("Train and Test", "train")
    if not os.path.exists(train_dir):
        print("Training directory not found.")
        return

    # Aggregate ground truth
    all_dfs = []
    for cls_dir in os.listdir(train_dir):
        gt_path = os.path.join(train_dir, cls_dir, "ground_truth.csv")
        if os.path.exists(gt_path):
            df = pd.read_csv(gt_path)
            df['folder_path'] = os.path.join(train_dir, cls_dir, "videos")
            all_dfs.append(df)
            
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Filter for timestamped events
    df['start_time_sec'] = pd.to_numeric(df['start_time_sec'], errors='coerce')
    df['end_time_sec'] = pd.to_numeric(df['end_time_sec'], errors='coerce')
    temporal_df = df.dropna(subset=['start_time_sec', 'end_time_sec']).copy()
    
    print(f"\n{'='*60}")
    print("TIMESTAMP VIOLATION SUMMARY REPORT")
    print(f"{'='*60}")
    
    reasons_count = defaultdict(int)
    end_diffs = []
    
    print("Scanning all timestamped training videos... (this will take a minute)")
    
    for _, row in temporal_df.iterrows():
        vid_id = row['video_id']
        start = row['start_time_sec']
        end = row['end_time_sec']
        
        vid_path = os.path.join(row['folder_path'], f"{vid_id}.mp4")
        dur = get_video_duration(vid_path)
        
        if dur is not None:
            # Categorize the violation
            if start < 0:
                reasons_count["Start < 0"] += 1
            elif end <= start:
                reasons_count["End <= Start (Zero/Negative Duration)"] += 1
            elif start > dur:
                reasons_count["Start > Video Duration"] += 1
            elif end > dur:
                reasons_count["End > Video Duration"] += 1
                end_diffs.append(end - dur)

    print("\n--- VIOLATION BREAKDOWN ---")
    if not reasons_count:
        print("  No violations found!")
    else:
        for reason, count in sorted(reasons_count.items()):
            print(f"  {reason.ljust(45)} : {count} cases")
        
    if end_diffs:
        print("\n--- END > DURATION METRICS ---")
        print(f"  Total such cases   : {len(end_diffs)}")
        print(f"  Minimum difference : {np.min(end_diffs):.3f} sec")
        print(f"  Average difference : {np.mean(end_diffs):.3f} sec")
        print(f"  Maximum difference : {np.max(end_diffs):.3f} sec")

if __name__ == "__main__":
    summarize_violations()