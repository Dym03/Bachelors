from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import time

model = YOLO(model="runs/detect/yolo11l.pt_100_000_n2_50_2025-03-23/weights/best.pt")

start = time.time()
results = model(source="/run/media/honzadymacek/Elements SE/Honza-Skola/Bachelors/Python/data/video/cam0_20241125_123106-007.avi", stream=True)

for r in results:
    if len(r.boxes) > 0:
        r.show()

end = time.time()
length = end - start
print("It took", length, "seconds!")
results[0].show()
#print(results)
