from ultralytics import YOLO
from PIL import Image

import time
#model = YOLO(model="runs/detect/yolo11l.pt_100_000_n2_50_2025-03-23/weights/best.pt")
model = YOLO(model="runs/detect/yolo11l.pt_Mapillary_100_2025-03-12/weights/best.pt")
# results = model("data/img/rychlosti.jpg")

start = time.time()
results = model("datasets/GTSDB/val/images/00054.png")


end = time.time()
length = end - start
print("It took", length, "seconds!")
results[0].show()
#print(results)
