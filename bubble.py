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

    show_filtered = False
    show_original = False

    min_size = 150
    max_size = 1700

    cv2.namedWindow('Display', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, original_frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        foreground_frame = backSub.apply(original_frame)
        contours, _ = cv2.findContours(foreground_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        overlay_frame = np.zeros_like(original_frame)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            filtered = area < min_size or area > max_size

            if filtered and not show_filtered:
                continue

            if filtered:
                color = filtered_color
            else:
                color = target_color

            cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), color, 1)

        if show_original:
            combined_frame = cv2.addWeighted(overlay_frame, 1, original_frame, 1, 0)
        else:
            foreground_frame_bgr = cv2.cvtColor(foreground_frame, cv2.COLOR_GRAY2BGR)
            combined_frame = cv2.addWeighted(overlay_frame, 1, foreground_frame_bgr, 1, 0)

        cv2.imshow('Display', combined_frame)

        key = cv2.waitKey(1)
        
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
    # video_path = "C:/Users/a.warman/Downloads/vlc-record-2025-06-18-11h57m52s-12P_PYA_XNA_CON_MLC-10_2022-U_001_22-09-29_01-13-44_000.mp4-.mp4"
    # video_path = "output_2.mp4"
    video_path = "C:/Users/a.warman/Downloads/12P_PYA_XNA MLC Bubbles.mp4" + "cropped.mp4"
    # crop_video(video_path, 30)
    process_video(video_path)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()