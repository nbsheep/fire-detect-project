"""live_preview.py — 用训练中的最新检查点生成检测预览页
================================================================
在 5 个实拍视频的固定帧上跑当前 best.pt,拼成网格图 + 自动刷新 HTML。
训练期间跑它只占 GPU 几秒钟,不影响训练。
用法: python scripts/live_preview.py
输出: runs/live/preview.jpg + runs/live/view.html
"""
import csv
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "runs" / "detect" / "runs" / "fire_yolo26s_plus"
LIVE = ROOT / "runs" / "live"
VIDEOS = sorted((ROOT.parent / "Desktop" / "video").glob("*.mp4"))
FRACS = (0.4, 0.7)          # 每个视频取 40% 和 70% 处的帧,固定不变方便对比
CONF = 0.25
CELL_W = 560
LIVE.mkdir(parents=True, exist_ok=True)


def rd_video_frame(path, frac):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac) if total > 0 else 0)
    ok, frame = cap.read()
    cap.release()
    return frame


def main():
    ep, mAP50 = 0, 0.0
    f = RUN / "results.csv"
    if f.exists():
        rows = list(csv.DictReader(f.open(encoding="utf-8")))
        if rows:
            ep = int(rows[-1]["epoch"])
            mAP50 = float(rows[-1]["metrics/mAP50(B)"])

    model = YOLO(RUN / "weights" / "best.pt")
    cells, n_hit = [], 0
    for vid in VIDEOS:
        for frac in FRACS:
            frame = rd_video_frame(vid, frac)
            if frame is None:
                continue
            r = model.predict(frame, conf=CONF, imgsz=640, device=0, verbose=False)[0]
            boxes = []
            if r.boxes is not None and len(r.boxes):
                for xyxy, cls, c in zip(r.boxes.xyxy.cpu().numpy(),
                                        r.boxes.cls.cpu().numpy().astype(int),
                                        r.boxes.conf.cpu().numpy()):
                    boxes.append((xyxy, cls, c))
            if boxes:
                n_hit += 1
            h, w = frame.shape[:2]
            s = CELL_W / w
            cell = cv2.resize(frame, (CELL_W, int(h * s)))
            for xyxy, cls, c in boxes:
                x1, y1, x2, y2 = [int(v * s) for v in xyxy]
                color = (0, 0, 255) if cls == 0 else (0, 255, 0)
                cv2.rectangle(cell, (x1, y1), (x2, y2), color, 2)
                cv2.putText(cell, f"{c:.2f}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            tag = vid.stem.split("_")[-2] + "_" + vid.stem.split("_")[-1]
            cv2.putText(cell, tag, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cells.append(cell)

    if not cells:
        print("没有可用帧")
        return
    rows_img = [np.hstack(cells[i:i + 2]) for i in range(0, len(cells), 2)]
    wmax = max(r.shape[1] for r in rows_img)
    rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, wmax - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(25, 25, 25)) for r in rows_img]
    grid = np.vstack(rows_img)
    cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tofile(LIVE / "preview.jpg")

    now = time.strftime("%H:%M:%S")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="60"><title>训练实况</title>
<style>body{{background:#111;color:#ddd;font-family:Segoe UI,sans-serif;margin:16px}}
h1{{font-size:20px}} .big{{font-size:34px;color:#ff9b3d;font-weight:bold}}
img{{width:100%;max-width:1200px;border-radius:8px}}</style></head><body>
<h1>烟雾模型训练实况 <span style="font-size:14px;color:#888">(每60秒自动刷新,预览每~12分钟更新)</span></h1>
<p>第 <span class="big">{ep}</span>/80 轮 &nbsp; mAP50=<span class="big">{mAP50:.3f}</span>
&nbsp; 预览帧检出 {n_hit}/{len(cells)} @conf{CONF} &nbsp; 更新于 {now}</p>
<img src="preview.jpg"><br><br>
<img src="curves.png"></body></html>"""
    (LIVE / "view.html").write_text(html, encoding="utf-8")
    cp = RUN / "results.png"
    if cp.exists():
        shutil.copyfile(cp, LIVE / "curves.png")
    print(f"[预览] 轮{ep} mAP50={mAP50:.3f} 帧检出 {n_hit}/{len(cells)} -> runs/live/view.html")


if __name__ == "__main__":
    main()
