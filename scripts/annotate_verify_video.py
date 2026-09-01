"""
annotate_verify_video.py — 给视频生成"实时检测+火势预测"的标注视频
================================================================
用法:
  python scripts/annotate_verify_video.py <视频路径> [输出路径] [预测秒数]

输出: 一个带检测框和火势趋势/方向/预测叠加的 mp4,直接用播放器打开看。
"""
import sys
import time
import cv2
from ultralytics import YOLO
from fire_trend import FireTrendPredictor, union_box
from rtsp_detect import draw_cn

SRC = sys.argv[1] if len(sys.argv) > 1 else "samples/ForestFire1.avi"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/verify/forestfire1_annotated.avi"
HORIZON = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
CONF = 0.25
FIRE_CLASS = 1


def make_writer(path, fps, size):
    """选一个"播放器都能开"的编码。

    实测这台机器的 OpenCV 没有 H.264 编码器(mp4 会回退成 mp4v,
    很多播放器打不开)。所以统一用 AVI + MJPG,兼容性最好。
    """
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)


def main():
    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        print(f"[错误] 打不开 {SRC}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(3)), int(cap.get(4))
    writer = make_writer(OUT, fps, (W, H))
    model = YOLO("models/best.pt")
    pred = FireTrendPredictor()
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        r = model.predict(frame, conf=CONF, device=0, verbose=False)[0]
        annotated = r.plot()

        fb = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            fb = [b for b, c in zip(xyxy, cls) if c == FIRE_CLASS]
        if fb:
            pred.update(union_box(fb), frame.shape, t)
        else:
            pred.note_no_fire(t)

        res = pred.analyze(predict_seconds=HORIZON)
        if res is not None:
            l1 = f"趋势:{res.trend} 方向:{res.direction}"
            l2 = f"{HORIZON:.0f}s后面积 {res.pred_delta_pct:+.0f}%"
            draw_cn(annotated, l1, (10, 10), 26, (0, 255, 255))
            draw_cn(annotated, l2, (10, 44), 22, (0, 255, 255))
            if idx % int(fps) == 0:
                print(f"[{t:.0f}s] {l1} | {l2}")
        else:
            draw_cn(annotated, "采样中...", (10, 10), 22, (180, 180, 180))

        writer.write(annotated)
        idx += 1
        if idx % 100 == 0:
            print(f"  已处理 {idx} 帧")
    cap.release()
    writer.release()
    print(f"[完成] 标注视频已保存: {OUT}")


if __name__ == "__main__":
    main()
