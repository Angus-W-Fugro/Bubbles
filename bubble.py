import cv2
from vidstab import VidStab

def main():
    video_path = "C:/Users/a.warman/Downloads/vlc-record-2025-06-18-11h57m52s-12P_PYA_XNA_CON_MLC-10_2022-U_001_22-09-29_01-13-44_000.mp4-.mp4"
    cap = cv2.VideoCapture(video_path)
    stabilizer = VidStab()
    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=2000)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    while True:
        ret, original_frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        stabilised_frame = stabilizer.stabilize_frame(input_frame=original_frame, border_size=50)
        fg_mask = backSub.apply(stabilised_frame)
        
        # Display the frame
        overlay_weighting = 0.8
        combined = cv2.addWeighted(stabilised_frame, 1 - overlay_weighting, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), overlay_weighting, 0)
        cv2.imshow('Combined Frame', combined)
        
       # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()