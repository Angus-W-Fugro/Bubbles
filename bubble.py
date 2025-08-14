import cv2
from vidstab import VidStab

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
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=32, history=500, detectShadows=False, )
    playback = int(1000 / (cap.get(cv2.CAP_PROP_FPS) * 0.75))
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, original_frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # stable_frame = stabilizer.stabilize_frame(input_frame=original_frame, border_size=50)
        foreground_frame = backSub.apply(original_frame)
        cv2.imshow('Frame', foreground_frame)
        key = cv2.waitKey(playback)

        # foreground_weighting = 1
        # background_weighting = 0.1
        # combined_frame = cv2.addWeighted(stable_frame, background_weighting, cv2.cvtColor(foreground_frame, cv2.COLOR_GRAY2BGR), foreground_weighting, 0)
        # cv2.imshow('Combined Frame', combined_frame)
        
       # Break the loop if 'Esc' key is pressed
        if key == 27:
            exit(0)

def main():
    # video_path = "C:/Users/a.warman/Downloads/vlc-record-2025-06-18-11h57m52s-12P_PYA_XNA_CON_MLC-10_2022-U_001_22-09-29_01-13-44_000.mp4-.mp4"
    # split_frames_into_6_videos(video_path)
    # return
    process_video("output_2.mp4")

    # for i in range(6):
    #     process_video(f"output_{i+1}.mp4")

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()