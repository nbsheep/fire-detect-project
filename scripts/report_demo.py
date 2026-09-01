import cv2
from ultralytics import YOLO
from reporter import InspectionReporter   # 同目录下的模块

WEIGHTS = "models/best.pt"
SOURCE  = "samples/test.mp4"    # 需自己放一个视频到此路径;或改 0=摄像头, 或 RTSP/RTMP 地址
CONF    = 0.35

def main():
    model = YOLO(WEIGHTS)
    reporter = InspectionReporter()
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        print(f"[错误] 打不开视频源:{SOURCE}")
        print("      samples/ 里目前只有 test.jpg,没有视频。")
        print("      请放一个视频到 samples/test.mp4,或把 SOURCE 改成 0(摄像头)。")
        return

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(frame, conf=CONF, device=0, verbose=False)
        r = results[0]
        annotated = r.plot()

        # 遍历这一帧检测到的每个目标
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]     # 'fire' 或 'smoke'
            conf   = float(box.conf[0])
            # 这里可接入真实 GPS;暂用 None
            reporter.log_event(annotated, label, conf, gps=None)

    cap.release()
    reporter.save_report()

if __name__ == "__main__":
    main()