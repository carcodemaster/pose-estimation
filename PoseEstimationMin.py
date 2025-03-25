import cv2
import mediapipe as mp # type: ignore
import time
import os

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

cap = cv2.VideoCapture('Videos/1.mp4')
pTime = 0

# Check if running in Docker (determine by environment variable)
in_docker = os.environ.get('QT_QPA_PLATFORM') == 'offscreen'

# If in Docker, set up video writer
if in_docker:
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output/pose_estimation_output.avi', fourcc, fps, (width, height))
    print("Running in Docker mode: saving output to output/pose_estimation_output.avi")

while True:
    success, img = cap.read()
    if not success:
        print("End of video or error reading frame")
        break
        
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    
    if in_docker:
        # Write frame to output video
        out.write(img)
        # Print progress every 100 frames
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 100 == 0:
            print(f"Processed frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}")
    else:
        # Display the frame if not in Docker
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
if in_docker:
    out.release()
    print("Processing complete. Output saved to output/pose_estimation_output.avi")
else:
    cv2.destroyAllWindows()