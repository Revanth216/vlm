import cv2
import time

def detect_candidate_windows(video_path, motion_threshold_pct=1.0, window_size_sec=4.0, stride_sec=2.0, fps_target=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}. Check if the file path is correct!")
        return [], 0.0

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 25.0 
        
    frame_skip = int(orig_fps / fps_target)
    if frame_skip == 0:
        frame_skip = 1
        
    prev_gray = None
    frame_count = 0
    
    candidate_windows = []
    motion_buffer = [] # Rolling buffer to hold (timestamp, motion_pct)
    
    frames_per_window = int(window_size_sec * fps_target)
    frames_per_stride = int(stride_sec * fps_target)

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_skip == 0:
            current_sec = frame_count / orig_fps
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_gray is not None:
                frame_delta = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
                
                motion_pixels = cv2.countNonZero(thresh)
                total_pixels = thresh.shape[0] * thresh.shape[1]
                motion_pct = (motion_pixels / total_pixels) * 100
                
                motion_buffer.append((current_sec, motion_pct))
                
                # When buffer reaches window size, evaluate and slide
                if len(motion_buffer) == frames_per_window:
                    window_start = motion_buffer[0][0]
                    window_end = motion_buffer[-1][0]
                    max_motion = max([m for _, m in motion_buffer])
                    
                    if max_motion >= motion_threshold_pct:
                        candidate_windows.append((window_start, window_end, max_motion))
                        
                    # Slide the window forward by the stride amount (pop oldest frames)
                    motion_buffer = motion_buffer[frames_per_stride:]
            
            prev_gray = gray

        frame_count += 1

    cap.release()
    elapsed_time = time.time() - start_time
    
    return candidate_windows, elapsed_time

if __name__ == "__main__":
    TEST_VIDEO = "Train and Test/test/videos/T021.mp4" 
    
    print(f"Running Stage 1 (Overlapping) on {TEST_VIDEO}...")
    windows, runtime = detect_candidate_windows(TEST_VIDEO, motion_threshold_pct=1.0)
    
    print("\n" + "="*40)
    print("STAGE 1 RESULTS")
    print("="*40)
    print(f"Total processing time: {runtime:.2f} seconds")
    print(f"Candidate Windows Found: {len(windows)}")
    for start, end, motion in windows:
        print(f"  - [{start:.1f}s -> {end:.1f}s] (Max Motion: {motion:.2f}%)")