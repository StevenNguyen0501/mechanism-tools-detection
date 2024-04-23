from ultralytics import YOLO

model = YOLO('runs/detect/train14/weights/best.pt')
results = model.predict('mechanical-ver3-2/train/images/Wrench-74-_JPEG_jpg.rf.09f87e669cd8d70509114e6555afe4a7.jpg', save = True)
print(results)


