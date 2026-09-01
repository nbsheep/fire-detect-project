from ultralytics import YOLO

model = YOLO("models/best.pt")

# save=True 会把画好框的图存到 runs/detect/predict/ 下
results = model.predict(
    source="samples/test.jpg",
    save=True,
    conf=0.25,        # 置信度阈值,低于它的框不显示;误检多就调高,漏检多就调低
    device=0,
)
for r in results:
    print("检测到的目标数:", len(r.boxes))