import pandas as pd
import numpy as np

def analyze_windows(csv_path="training_windows.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        return

    total_windows = len(df)
    total_padded = df['needs_padding'].sum() if 'needs_padding' in df.columns else 0

    print(f"\n{'='*110}")
    print("TRAINING WINDOWS DIAGNOSTIC REPORT")
    print(f"{'='*110}")
    print(f"Total Windows Generated : {total_windows}")
    print(f"Windows Needing Padding : {total_padded} (Short events/videos)")
    print("-" * 110)

    # Group by label and video_id to calculate windows per video
    video_counts = df.groupby(['label', 'video_id']).size().reset_index(name='window_count')

    stats = []
    for label, group in video_counts.groupby('label'):
        num_videos = len(group)
        total_wins = group['window_count'].sum()
        
        # Calculate statistics
        min_w = group['window_count'].min()
        p25_w = np.percentile(group['window_count'], 25)
        med_w = group['window_count'].median()
        p75_w = np.percentile(group['window_count'], 75)
        max_w = group['window_count'].max()
        mean_w = group['window_count'].mean()

        stats.append({
            'Class': label,
            'Videos': num_videos,
            'Windows': total_wins,
            'Min': min_w,
            'P25': p25_w,
            'Median': med_w,
            'P75': p75_w,
            'Max': max_w,
            'Mean': mean_w
        })

    # Sort by total windows descending
    stats.sort(key=lambda x: x['Windows'], reverse=True)

    # Formatting and printing
    format_str = "{:<32} | {:<8} | {:<9} | {:<5} | {:<5} | {:<6} | {:<5} | {:<5} | {:<6}"
    print(format_str.format("CLASS", "VIDEOS", "WINDOWS", "MIN", "P25", "MEDIAN", "P75", "MAX", "MEAN"))
    print("-" * 110)

    for s in stats:
        print(format_str.format(
            s['Class'], 
            s['Videos'], 
            s['Windows'],
            s['Min'], 
            f"{s['P25']:.1f}", 
            f"{s['Median']:.1f}",
            f"{s['P75']:.1f}", 
            s['Max'], 
            f"{s['Mean']:.1f}"
        ))
    print("=" * 110)

if __name__ == "__main__":
    analyze_windows()