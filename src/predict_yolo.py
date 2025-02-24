from ultralytics import YOLO
from PIL import Image

model = YOLO(model="models/yolo_models/yolo_v11m_50e_10_000_n2.pt")

# results = model("data/img/rychlosti.jpg")
image = Image.open("00515.ppm").save("karel.png")

results = model("karel.png")

results[0].show()
