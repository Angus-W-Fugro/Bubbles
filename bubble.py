import cv2
from vidstab import VidStab
import numpy as np

def main():
    video_path = "C:/Users/a.warman/Downloads/vlc-record-2025-06-18-11h57m52s-12P_PYA_XNA_CON_MLC-10_2022-U_001_22-09-29_01-13-44_000.mp4-.mp4"
    cap = cv2.VideoCapture(video_path)
    stabilizer = VidStab()
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=128, history=500, detectShadows=False)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    color = np.random.randint(0, 255, (100, 3))

    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    while True:
        ret, original_frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        stable_frame = stabilizer.stabilize_frame(input_frame=original_frame, border_size=50)
        foreground_frame = backSub.apply(stable_frame)

        frame_gray = cv2.cvtColor(stable_framez, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        labelled_frame = cv2.add(foreground_frame, mask)

        cv2.imshow('Frame', labelled_frame)


        # foreground_weighting = 1
        # background_weighting = 0.1
        # combined_frame = cv2.addWeighted(stable_frame, background_weighting, cv2.cvtColor(foreground_frame, cv2.COLOR_GRAY2BGR), foreground_weighting, 0)
        # cv2.imshow('Combined Frame', combined_frame)
        
       # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()