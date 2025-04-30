import cv2
from ultralytics import YOLO


# model = YOLO(model="runs/detect/yolo11l.pt_Mapillary_100_2025-03-12/weights/best.pt")
# model = YOLO(model="runs/detect/yolo11l.pt_100_000_n2_50_2025-03-23/weights/best.pt")
model = YOLO(model="runs/detect/train6/weights/best.pt")

cap = cv2.VideoCapture("/run/media/honzadymacek/Elements SE/Honza-Skola/Bachelors/Python/data/video/transfer_239148_files_f8011902/cam0_20241125_105954.avi")
# cap = cv2.VideoCapture("/home/honzadymacek/Documents/Python/traffic_lights/data/video/2018_1106_062930_015F.MP4")

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
            img_w, img_h = flipped_frame.shape[1], flipped_frame.shape[0]  # Get frame dimensions
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
