from ultralytics import YOLO

# # Load a pre-trained YOLOv10n model
# model = YOLO("yolov10x.pt")

# results = model("src/data/img/stop_2.jpg")

# results[0].show()

model = YOLO("yolov10n.pt")

results = model.train(data="datasets/yolo_dataset_2/dataset.yaml", epochs=10, imgsz=512)
