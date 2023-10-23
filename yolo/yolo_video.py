from ultralytics import YOLO

model = YOLO("yolov8x.pt")
results = model.predict("../img/video_V.mp4", classes=[60,67,73,76], save=True)
