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
    
    bubble_color = (0, 0, 255)
    min_size = 5
    max_size = 100

    while True:
        ret, original_frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        foreground_frame = backSub.apply(original_frame)
        contours, _ = cv2.findContours(foreground_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_size or area > max_size:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            radius = max(w, h) // 2
            cv2.circle(original_frame, center, radius, bubble_color, 2)
            
        foreground_frame = original_frame

        cv2.imshow('Frame', foreground_frame)
        key = cv2.waitKey(100)
        
        if key == 27: # esc
            exit(0)

def main():
    video_path = "C:/Users/a.warman/Downloads/vlc-record-2025-06-18-11h57m52s-12P_PYA_XNA_CON_MLC-10_2022-U_001_22-09-29_01-13-44_000.mp4-.mp4"
    # video_path = "output_2.mp4"
    process_video(video_path)

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()