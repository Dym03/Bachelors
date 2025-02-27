from ultralytics import YOLO

# # Load a pre-trained YOLOv10n model
# model = YOLO("yolov10x.pt")

# results = model("src/data/img/stop_2.jpg")

# results[0].show()

model = YOLO("yolo11l.pt")

results = model.train(data="datasets/100_000/dataset.yaml", epochs=100, imgsz=512)
