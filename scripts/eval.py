from ultralytics import YOLO

model = YOLO("models/best.pt")

if __name__ == "__main__":
    metrics = model.val(data="data/dfire.yaml", imgsz=640, device=0)
    print("mAP50    :", round(metrics.box.map50, 4))   # 越接近 1 越好
    print("mAP50-95 :", round(metrics.box.map, 4))