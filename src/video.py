import cv2
import sys
import os
from ultralytics import YOLO

MODEL_PATH = "models/yolo_models/yolo11l.pt_CATSD_50_2025-03-23.pt"
VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else "" # Either specify as argument or specify here
if VIDEO_PATH == "":
    print("Please specify Video Path as a first argument")
    exit(-1)
elif not os.path.exists(VIDEO_PATH):
    print("Please specify a valid video path as a first argument")
    exit(-1)
    
model = YOLO(model=MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# skip_seconds = 5 * 60 + 15
# frame_to_start = skip_seconds * fps

# # Seek to the specific frame
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_start)
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame (0 = vertical, 1 = horizontal, -1 = both)
        flipped_frame = cv2.flip(frame, -1)  # Change 1 to 0 or -1 if needed
        # flipped_frame = frame
        results = model(flipped_frame, verbose=False)
        
        for result in results[0].boxes:  # loop through each detection
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            class_id, conf = result.cls.item(), result.conf.item()

            # Convert YOLO box format (center_x, center_y, width, height) to (top-left, bottom-right)
            img_w, img_h = flipped_frame.shape[1], flipped_frame.shape[0]
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)

            # Get the label and confidence score
            label = f"{model.names[int(class_id)]} {conf:.2f}"

            # Draw the bounding box and label
            cv2.rectangle(flipped_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle (BGR color)
            cv2.putText(flipped_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
    cv2.imshow("Flipped Video", flipped_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):  # Quit on 'q'
        break
    elif key == 32:  # Spacebar toggles pause
        paused = not paused

# Release resources
cap.release()
cv2.destroyAllWindows()
