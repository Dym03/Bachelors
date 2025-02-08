from ultralytics import YOLO


model = YOLO(model="models/yolo_models/yolo_v11l_50e_10_000_n2.pt")

results = model("data/img/rychlosti.jpg")

results[0].show()