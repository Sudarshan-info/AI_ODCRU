from ultralytics import YOLO

model = YOLO("models/license.pt")
print(model.names)
