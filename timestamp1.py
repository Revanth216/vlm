import os
import cv2
import pandas as pd
import numpy as np
from collections import defaultdict

# 11 anomaly classes + normal
EXPECTED_CLASSES = {
    "traffic_accident", "traffic_congestion", "stalled_or_broken_down_vehicle", 
    "vehicle_blocking_traffic", "wrong_way_driving", "road_spill_or_debris", 
    "waterlogging_or_flood", "fire", "smoke", "fighting_or_violence", 
    "loitering_or_suspicious_presence", "normal"
}

def analyze_video_metadata(video_path):
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = fc / fps if fps > 0 else 0
    return {"fps": fps, "width": w, "height": h, "frames": fc, "duration": dur}

def scan_directory(data_dir, is_train=False):
    title = "TRAINING DATA" if is_train else "PUBLIC TEST"
    print(f"\n{'='*60}\nAHC DATASET INSPECTION: {title}\n{'='*60}")
    
    gt_dfs, vid_dfs, mp4_files = [], [], set()
    
    # 1. Aggregate Data & MP4s
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))] if is_train else [data_dir]
    
    for d in subdirs:
        gt_path = os.path.join(d, "ground_truth.csv")
        vid_path = os.path.join(d, "videos.csv")
        v_folder = os.path.join(d, "videos")
        
        if os.path.exists(gt_path): gt_dfs.append(pd.read_csv(gt_path))
        if os.path.exists(vid_path): vid_dfs.append(pd.read_csv(vid_path))
        if os.path.exists(v_folder):
            for f in os.listdir(v_folder):
                if f.endswith('.mp4'):
                    mp4_files.add(f.split('.')[0])
                    
    if not gt_dfs:
        print("No ground_truth.csv files found.")
        return
        
    df = pd.concat(gt_dfs, ignore_index=True)
    v_df = pd.concat(vid_dfs, ignore_index=True) if vid_dfs else pd.DataFrame()
    
    # 2. Extract Metadata for ALL physical videos
    print("Extracting physical video metadata... (this may take a moment)")
    video_metadata = {}
    fps_dist, res_dist, durations = defaultdict(int), defaultdict(int), []
    
    for d in subdirs:
        v_folder = os.path.join(d, "videos")
        if os.path.exists(v_folder):
            for f in os.listdir(v_folder):
                if f.endswith('.mp4'):
                    vid_id = f.split('.')[0]
                    meta = analyze_video_metadata(os.path.join(v_folder, f))
                    if meta:
                        video_metadata[vid_id] = meta
                        fps_dist[f"{round(meta['fps'])} FPS"] += 1
                        res_dist[f"{meta['width']}x{meta['height']}"] += 1
                        durations.append(meta['duration'])

    # --- 1. How much data do we have? ---
    print(f"\n1. HOW MUCH DATA DO WE HAVE?")
    print(f"  Total Unique Videos (Ground Truth): {df['video_id'].nunique()}")
    
    # --- 2. Class Distribution ---
    print(f"\n2. CLASS DISTRIBUTION:")
    for cls, count in df['class_name'].value_counts().items():
        print(f"  {cls.ljust(35)}: {count}")
        
    # --- 3. Events Per Video (Including 0 events) ---
    print(f"\n3. EVENTS PER VIDEO:")
    # Count only anomaly classes per video
    event_counts = df.groupby('video_id').apply(lambda x: (x['class_name'] != 'normal').sum())
    for count, videos in event_counts.value_counts().sort_index().items():
        print(f"  {count} event(s): {videos} videos")

    # --- 4. Level Distribution ---
    print(f"\n4. LEVEL DISTRIBUTION:")
    level_conflicts = 0
    if 'level' in df.columns:
        level_groups = df.groupby('video_id')['level'].unique()
        level_conflicts = sum(len(lvls) > 1 for lvls in level_groups)
        levels = df.groupby('video_id')['level'].first().value_counts()
        for lvl, count in sorted(levels.items()):
            print(f"  Level {lvl}: {count} videos")
        if level_conflicts > 0:
            print(f"  [WARNING] {level_conflicts} videos have conflicting level annotations!")
    else:
        print("  Level data missing.")

    # --- 5. Temporal Distribution ---
    print(f"\n5. TEMPORAL DIFFICULTY (Event Durations):")
    temporal_df = df.copy()
    temporal_df['start_time_sec'] = pd.to_numeric(temporal_df['start_time_sec'], errors='coerce')
    temporal_df['end_time_sec'] = pd.to_numeric(temporal_df['end_time_sec'], errors='coerce')
    
    valid_temporal = temporal_df.dropna(subset=['start_time_sec', 'end_time_sec']).copy()
    if not valid_temporal.empty:
        valid_temporal['duration'] = valid_temporal['end_time_sec'] - valid_temporal['start_time_sec']
        durs = valid_temporal['duration'].values
        print(f"  Shortest : {np.min(durs):.2f} sec")
        print(f"  P25      : {np.percentile(durs, 25):.2f} sec")
        print(f"  Median   : {np.median(durs):.2f} sec")
        print(f"  P75      : {np.percentile(durs, 75):.2f} sec")
        print(f"  Longest  : {np.max(durs):.2f} sec")
    else:
        print("  No valid timestamps found.")

    # --- 6. Video Metadata ---
    print(f"\n6. VIDEO METADATA:")
    print("  FPS:")
    for k, v in sorted(fps_dist.items()): print(f"    {k}: {v}")
    print("  Resolution:")
    for k, v in sorted(res_dist.items()): print(f"    {k}: {v}")
    if durations:
        print("  Video Durations:")
        print(f"    Min: {np.min(durations):.1f}s | Median: {np.median(durations):.1f}s | Max: {np.max(durations):.1f}s")

    # --- 7. Data Integrity Checks ---
    print(f"\n7. DATA INTEGRITY:")
    
    # Class Check
    unexpected = set(df['class_name'].unique()) - EXPECTED_CLASSES
    print(f"  [{'✓' if not unexpected else '✗'}] Class names valid {f'(Found unexpected: {unexpected})' if unexpected else ''}")
    
    # CSV vs MP4 Check
    gt_ids = set(df['video_id'].unique())
    vid_ids = set(v_df['video_id'].unique()) if not v_df.empty else gt_ids
    
    missing_mp4s = gt_ids - mp4_files
    print(f"  [{'✓' if not missing_mp4s else '✗'}] All GT videos have MP4 files {f'({len(missing_mp4s)} missing)' if missing_mp4s else ''}")
    print(f"  [{'✓' if gt_ids == vid_ids else '✗'}] ground_truth.csv matches videos.csv")

    # Timestamp Validation
    bad_times = 0
    if not valid_temporal.empty:
        for _, row in valid_temporal.iterrows():
            vid = row['video_id']
            start, end = row['start_time_sec'], row['end_time_sec']
            vid_dur = video_metadata.get(vid, {}).get('duration', float('inf'))
            
            if start < 0 or end <= start or start > vid_dur or end > vid_dur:
                bad_times += 1
                
    print(f"  [{'✓' if bad_times == 0 else '✗'}] Timestamps fit within video durations {f'({bad_times} violations)' if bad_times > 0 else ''}")

if __name__ == "__main__":
    train_dir = os.path.join("Train and Test", "train")
    test_dir = os.path.join("Train and Test", "test")
    
    if os.path.exists(train_dir): scan_directory(train_dir, is_train=True)
    if os.path.exists(test_dir): scan_directory(test_dir, is_train=False)