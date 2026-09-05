import os
import cv2
import pandas as pd
import numpy as np

WINDOW_SIZE = 4.0
STRIDE = 2.0
NORMAL_WINDOWS_PER_VIDEO = 3  # Normal training samples (baseline)

def get_video_duration(video_path):
    if not os.path.exists(video_path): return 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    fc = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fc / fps if fps > 0 else 0

def generate_windows():
    train_dir = os.path.join("Train and Test", "train")
    all_windows = []
    
    print("Generating temporal training windows...")
    
    for cls_dir in os.listdir(train_dir):
        gt_path = os.path.join(train_dir, cls_dir, "ground_truth.csv")
        vid_dir = os.path.join(train_dir, cls_dir, "videos")
        
        if not os.path.exists(gt_path): continue
        df = pd.read_csv(gt_path)
        
        for _, row in df.iterrows():
            vid = row['video_id']
            cls_name = row['class_name']
            vid_path = os.path.join(vid_dir, f"{vid}.mp4")
            
            if not os.path.exists(vid_path): continue
            dur = get_video_duration(vid_path)
            if dur <= 0: continue 
            
            # --- NORMAL VIDEOS (Training Samples) ---
            if cls_name == "normal":
                if dur < WINDOW_SIZE:
                    all_windows.append({
                        "video_id": vid, "video_path": vid_path,
                        "window_start": 0.0, "window_end": round(dur, 2),
                        "label": "normal", "event_start": None, "event_end": None,
                        "needs_padding": True
                    })
                else:
                    max_start = dur - WINDOW_SIZE
                    starts = np.linspace(0, max_start, num=NORMAL_WINDOWS_PER_VIDEO)
                    for s in starts:
                        all_windows.append({
                            "video_id": vid, "video_path": vid_path,
                            "window_start": round(s, 2), "window_end": round(s + WINDOW_SIZE, 2),
                            "label": "normal", "event_start": None, "event_end": None,
                            "needs_padding": False
                        })
                continue
                
            # --- ANOMALY VIDEOS ---
            start = pd.to_numeric(row['start_time_sec'], errors='coerce')
            end = pd.to_numeric(row['end_time_sec'], errors='coerce')
            
            if pd.isna(start) or pd.isna(end): continue
            
            # 1. Apply safety clipping (Unconditional clip to duration)
            usable_start = start
            usable_end = min(end, dur)
            
            # Drop invalid or negative-duration windows
            if usable_end <= usable_start: continue
            
            # 2. Generate Sliding Windows
            event_duration = usable_end - usable_start
            
            if event_duration < WINDOW_SIZE:
                # Center a single 4s window over short events
                center = usable_start + (event_duration / 2.0)
                w_start = max(0, center - (WINDOW_SIZE / 2.0))
                w_end = min(dur, w_start + WINDOW_SIZE)
                
                # If we still can't hit 4 seconds, mark it for padding
                needs_pad = (w_end - w_start) < (WINDOW_SIZE - 0.05)
                
                all_windows.append({
                    "video_id": vid, "video_path": vid_path,
                    "window_start": round(w_start, 2), "window_end": round(w_end, 2),
                    "label": cls_name, "event_start": usable_start, "event_end": usable_end,
                    "needs_padding": needs_pad
                })
            else:
                # Slide across longer events
                curr = usable_start
                while (curr + WINDOW_SIZE) <= usable_end:
                    all_windows.append({
                        "video_id": vid, "video_path": vid_path,
                        "window_start": round(curr, 2), "window_end": round(curr + WINDOW_SIZE, 2),
                        "label": cls_name, "event_start": usable_start, "event_end": usable_end,
                        "needs_padding": False
                    })
                    curr += STRIDE
                
                # Ensure we capture the tail of the event
                last_added_end = (curr - STRIDE) + WINDOW_SIZE
                if last_added_end < usable_end:
                    tail_start = max(0, usable_end - WINDOW_SIZE)
                    needs_pad = (usable_end - tail_start) < (WINDOW_SIZE - 0.05)
                    all_windows.append({
                        "video_id": vid, "video_path": vid_path,
                        "window_start": round(tail_start, 2), "window_end": round(usable_end, 2),
                        "label": cls_name, "event_start": usable_start, "event_end": usable_end,
                        "needs_padding": needs_pad
                    })
                    
    out_df = pd.DataFrame(all_windows)
    out_df.to_csv("training_windows.csv", index=False)
    
    print(f"\nGenerated {len(out_df)} training windows.")
    print("\nWindow Label Distribution:")
    print(out_df['label'].value_counts())

if __name__ == "__main__":
    generate_windows()