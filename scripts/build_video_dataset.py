"""build_video_dataset.py — 从实拍视频抽帧,用现有模型伪标注,生成 YOLO 训练数据
================================================================
用法:
  python scripts/build_video_dataset.py <视频目录或视频路径>... [--stride 10] [--conf 0.30]

输出: data/video_frames/{images,labels}/  (YOLO 格式,smoke=0, fire=1)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def pink_mask(frame):
    """彩烟(粉/红)像素掩码。烟饼视频里烟色和草地/天空区分度很高。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    low = ((h <= 8) | (h >= 172)) & (s >= 25) & (s <= 160) & (v >= 110)
    return low.astype(np.uint8)


def pink_box(frame):
    """有显著粉色区域时返回其外接框 (x1,y1,x2,y2 像素),否则 None。"""
    m = pink_mask(frame)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    area = frame.shape[0] * frame.shape[1]
    best = None
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < area * 0.002:
            continue
        best = (x, y, x + w, y + h) if best is None else (
            min(best[0], x), min(best[1], y), max(best[2], x + w), max(best[3], y + h))
    return best

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "video_frames"

STRIDE = 10
# 伪标注阈值策略:
#   >= KEEP_CONF 的框才写进标签;
#   只检出 AMBIG_CONF~KEEP_CONF 之间弱框的帧整个丢弃(旧模型不确定,标注会带偏见);
#   连弱框都没有的帧保留为负样本(空标签)。
KEEP_CONF = 0.15
AMBIG_CONF = 0.05
sources = [Path(a) for a in sys.argv[1:]] or [ROOT / "samples"]

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

model = YOLO(ROOT / "models" / "best.pt")

videos = []
for s in sources:
    if s.is_dir():
        videos.extend(sorted(s.glob("*.mp4")) + sorted(s.glob("*.avi")))
    else:
        videos.append(s)

total_img, total_box = 0, 0
for vid in videos:
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        print(f"[跳过] 打不开 {vid}")
        continue
    stem = vid.stem
    idx, kept, boxes, dropped, rescued = 0, 0, 0, 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STRIDE != 0:
            idx += 1
            continue
        r = model.predict(frame, conf=AMBIG_CONF, device=0, verbose=False)[0]
        lines = []
        model_kept_any = False
        if r.boxes is not None and len(r.boxes):
            confs = r.boxes.conf.cpu().numpy()
            H, W = frame.shape[:2]
            for xyxy, cls, c in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                confs,
            ):
                if c >= KEEP_CONF:
                    model_kept_any = True
                    x1, y1 = max(xyxy[0], 0) / W, max(xyxy[1], 0) / H
                    x2, y2 = min(xyxy[2], W) / W, min(xyxy[3], H) / H
                    lines.append(f"{cls} {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}")
        if not model_kept_any:
            # 模型没把握时用粉色掩码兜底: 掩码显著 -> 生成 smoke 框; 不显著 -> 丢弃
            pb = pink_box(frame)
            if pb is None:
                dropped += 1
                idx += 1
                continue
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = pb
            lines = [f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}"]
            rescued += 1
        name = f"{stem}_f{idx:06d}"
        cv2.imwrite(str(OUT / "images" / f"{name}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / "labels" / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        kept += 1
        boxes += len(lines)
        idx += 1
    cap.release()
    print(f"{stem}: 抽 {kept} 帧, {boxes} 个框, 掩码救回 {rescued}, 丢弃 {dropped}")
    total_img += kept
    total_box += boxes

print(f"[完成] 共 {total_img} 帧 / {total_box} 个框 -> {OUT}")
