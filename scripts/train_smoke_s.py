"""train_smoke_s.py — 用 D-Fire + 实拍视频伪标注数据训练 yolo26s
================================================================
比原 nano 大 4 倍容量,且训练集里混入了真实部署域(彩烟航拍)的帧。
用法: python scripts/train_smoke_s.py
"""
from ultralytics import YOLO

model = YOLO("yolo26s.pt")

if __name__ == "__main__":
    model.train(
        data="data/dfire_plus.yaml",
        epochs=80,
        patience=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project="runs",
        name="fire_yolo26s_plus",
    )
