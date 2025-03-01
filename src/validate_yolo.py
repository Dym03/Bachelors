from ultralytics import YOLO

MODEL_PATH = "yolo11l.pt"

model = YOLO(MODEL_PATH)

metrics = model.val(data="datasets/100_000_n2/dataset.yaml", imgsz=512)

print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps) # list of mAP50-95 for each category



