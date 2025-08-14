import cv2

def main():
    video_path = "C:/Users/a.warman/Downloads/vlc-record-2025-06-18-11h57m52s-12P_PYA_XNA_CON_MLC-10_2022-U_001_22-09-29_01-13-44_000.mp4-.mp4"
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        fg_mask = backSub.apply(frame)
        
        # Display the frame

        overlay_weighting = 0.8
        combined = cv2.addWeighted(frame, 1 - overlay_weighting, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), overlay_weighting, 0)
        cv2.imshow('Combined Frame', combined)
        
        # Wait for 1 ms and check if 'q' is pressed to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()