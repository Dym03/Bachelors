from ultralytics import YOLO
import sys
from datetime import date
# # Load a pre-trained YOLOv10n model
# model = YOLO("yolov10x.pt")

# results = model("src/data/img/stop_2.jpg")

# results[0].show()
NUM_EPOCHS = 100
DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else "Mapillary"
MODEL_DIR = "runs/detect/yolo11l.pt_Mapillary_100_2025-03-12/weights"
MODEL_NAME = "last.pt"
model = YOLO(f"{MODEL_DIR}/{MODEL_NAME}")
today_date = date.today().isoformat()
run_name = f"{MODEL_NAME}_{DATASET_NAME}_{NUM_EPOCHS}_{today_date}"

results = model.train(data=f"datasets/{DATASET_NAME}/dataset.yaml", epochs=NUM_EPOCHS, imgsz=1024, name=run_name, device=[2, 3], batch=0.8, resume=True, fliplr=0.0)
