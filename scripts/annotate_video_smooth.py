"""annotate_video_smooth.py — 烟雾检测标注视频(带时间平滑,消除框闪烁)
================================================================
用法:
  python scripts/annotate_video_smooth.py <视频路径> [输出路径] [模型路径]

原理: 逐帧独立检测时,弥散烟雾的置信度会在阈值上下抖动,框忽隐忽现。
这里用滞后锁定(hysteresis):
  - 新目标须 conf >= CONF_HI 才激活;
  - 已激活目标只要 conf >= CONF_LO 就维持(IoU 匹配延续);
  - 连续 MAX_MISS 帧匹配不上才移除,期间沿用上一次的框。
输出: 带平滑后检测框和烟雾状态行的 mp4/avi。
"""
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from rtsp_detect import draw_cn

ROOT = Path(__file__).resolve().parent.parent
SRC = sys.argv[1] if len(sys.argv) > 1 else "samples/ForestFire1.avi"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/verify/smooth_annotated.avi"
MODEL = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "models" / "best.pt"

CONF_HI = 0.20   # 新目标激活阈值
CONF_LO = 0.05   # 存活维持阈值
MAX_MISS = 10    # 连续丢失多少帧后移除
IOU_THR = 0.3

CLS_CN = {0: "烟", 1: "火"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    cap = cv2.VideoCapture(SRC)
    if not cap.isOpened():
        print(f"[错误] 打不开 {SRC}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    model = YOLO(MODEL)

    active = []  # {cls, box, conf, miss}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, conf=CONF_LO, verbose=False, device=0)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            for xyxy, cls, c in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy(),
            ):
                dets.append((cls, xyxy.tolist(), float(c)))

        matched = set()
        for a in active:
            best, best_iou = None, IOU_THR
            for j, (cls, box, c) in enumerate(dets):
                if j in matched or cls != a["cls"] or c < CONF_LO:
                    continue
                v = iou(a["box"], box)
                if v > best_iou:
                    best, best_iou = j, v
            if best is not None:
                matched.add(best)
                cls, box, c = dets[best]
                a["box"] = box
                a["conf"] = c
                a["miss"] = 0
            else:
                a["miss"] += 1
        active = [a for a in active if a["miss"] <= MAX_MISS]

        # 新目标激活: 高置信度且未匹配到任何已有目标
        for j, (cls, box, c) in enumerate(dets):
            if j in matched or c < CONF_HI:
                continue
            if not any(a["cls"] == cls and iou(a["box"], box) > IOU_THR for a in active):
                active.append({"cls": cls, "box": box, "conf": c, "miss": 0})

        vis = frame.copy()
        n_smoke = sum(1 for a in active if a["cls"] == 0)
        n_fire = sum(1 for a in active if a["cls"] == 1)
        for a in active:
            x1, y1, x2, y2 = map(int, a["box"])
            fade = max(0.35, 1 - a["miss"] / (MAX_MISS + 1))
            color = (0, 0, 255) if a["cls"] == 0 else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            draw_cn(vis, f"{CLS_CN[a['cls']]} {a['conf']:.2f}", (x1, max(y1 - 30, 0)),
                    20, color)
        if n_fire:
            draw_cn(vis, f"检测到火 {n_fire} 处 烟 {n_smoke} 处", (10, 10), 30, (0, 0, 255))
        elif n_smoke:
            draw_cn(vis, f"烟雾 {n_smoke} 处", (10, 10), 30, (0, 200, 255))
        else:
            draw_cn(vis, "无烟雾", (10, 10), 30, (180, 180, 180))

        writer.write(vis)
        idx += 1
        if idx % 100 == 0:
            print(f"  已处理 {idx} 帧")
    cap.release()
    writer.release()
    print(f"[完成] 已保存: {OUT}")


if __name__ == "__main__":
    main()
