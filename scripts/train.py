from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    model.train(data='C:/Users/odezz/source/MinecraftAI2/yolov8/dataset.yaml', epochs=50, imgsz=640, project='../runs', name='exp', device=0, workers=0, amp=False)
