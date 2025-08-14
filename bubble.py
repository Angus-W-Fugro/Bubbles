import cv2
from vidstab import VidStab
import numpy as np

def split_frames_into_6_videos(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rect_w = width // 3
    rect_h = height // 2

    outs = []
    for i in range(6):
        out_path = f"output_{i+1}.mp4"
        outs.append(cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (rect_w, rect_h)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for i in range(6):
            row = 0 if i < 3 else 1
            col = i % 3

            y_start = row * rect_h
            y_end = (row + 1) * rect_h
            x_start = col * rect_w
            x_end = (col + 1) * rect_w
                
            rect = frame[y_start:y_end, x_start:x_end]
            outs[i].write(rect)

    for out in outs:
        out.release()
    cap.release()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stabilizer = VidStab()
    threshold = 100
    history = 500

    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=threshold, history=history, detectShadows=False)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    target_color = (0, 0, 255)
    filtered_color = (0, 255, 0)

    show_size_filtered = False
    show_delta_filtered = False
    show_original = True
    frame_delay = 10

    min_size = 7
    max_size = 30

    cv2.namedWindow('Display', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_rects = []

    bubble_diameter_sum_per_frame = []

    while True:
        ret, original_frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        foreground_frame = backSub.apply(original_frame)
        contours, _ = cv2.findContours(foreground_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay_frame = np.zeros_like(original_frame)

        rects = []

        bubble_diameter_sum = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            max_dimension = max(w, h)
            min_dimension = min(w, h)

            size_filtered = min_dimension < min_size or max_dimension > max_size

            if not show_size_filtered and size_filtered:
                continue

            rects.append((x, y, w, h))

            delta_filtered = False

            # Find the closest rectangle in prev_rects to the current rect
            if prev_rects:
                distances = [np.hypot(x - px, y - py) for px, py, pw, ph in prev_rects]
                min_dist_idx = int(np.argmin(distances))
                closest_rect = prev_rects[min_dist_idx]
            else:
                closest_rect = None

            if closest_rect:
                cx, cy = x + w // 2, y + h // 2
                pcx, pcy = closest_rect[0] + closest_rect[2] // 2, closest_rect[1] + closest_rect[3] // 2
                dx = pcx - cx 
                dy = pcy - cy

                if dy < 5 or abs(dx) > abs(dy):
                    delta_filtered = True

            if delta_filtered:
                color = filtered_color
            else:
                color = target_color

            if show_delta_filtered or not delta_filtered:
                cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), color, 2)

            if not delta_filtered:
                diameter = (w + h) / 2
                bubble_diameter_sum += diameter
        
        prev_rects = rects

        if show_original:
            combined_frame = cv2.addWeighted(overlay_frame, 1, original_frame, 1, 0)
        else:
            foreground_frame_bgr = cv2.cvtColor(foreground_frame, cv2.COLOR_GRAY2BGR)
            combined_frame = cv2.addWeighted(overlay_frame, 1, foreground_frame_bgr, 1, 0)

        bubble_diameter_sum_per_frame.append(bubble_diameter_sum)
        fps = 30
        window_size_seconds = 1
        window_size = fps * window_size_seconds
        moving_average = np.mean(bubble_diameter_sum_per_frame[-window_size:]) if len(bubble_diameter_sum_per_frame) > window_size else bubble_diameter_sum

        text = f"Bubble diameter (px): {int(moving_average)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        color = (0, 0, 255)  # Red in BGR

        cv2.putText(combined_frame, text, (20, 900), font, font_scale, color, thickness, cv2.LINE_AA)

        cv2.imshow('Display', combined_frame)

        key = cv2.waitKey(frame_delay)
        
        if key == 27: # esc
            exit(0)

def crop_video(video_path, from_seconds):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(from_seconds * fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path + 'cropped.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()

def main():
    video_path = "C:/Users/a.warman/Downloads/vlc-record-2025-06-18-11h57m52s-12P_PYA_XNA_CON_MLC-10_2022-U_001_22-09-29_01-13-44_000.mp4-.mp4"
    # video_path = "output_2.mp4"
    # video_path = "C:/Users/a.warman/Downloads/12P_PYA_XNA MLC Bubbles.mp4" + "cropped.mp4"
    # crop_video(video_path, 30)
    process_video(video_path)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()