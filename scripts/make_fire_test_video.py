"""
make_fire_test_video.py — 生成"理想条件"的火焰测试视频
=====================================================
从真实火焰视频(ForestFire1.avi)里抠出火焰纹理,放到固定背景上,
让它面积随时间线性增长、中心位置固定、镜头静止。

这样我们有了"已知真值"的受控视频:火就是匀速变大。用它跑
fire_trend_verify.py,就能验证"预测方法本身准不准"——
如果误差小,说明方法没问题,之前 ForestFire1 误差大纯粹是
镜头移动/抖动违反了方法的前提假设。

用法:
  python scripts/make_fire_test_video.py
  python scripts/fire_trend_verify.py samples/synthetic_fire_grow.avi 5.0
"""
import cv2
import numpy as np

SRC = r"C:/Users/nice/Downloads/ForestFire1.avi"
OUT = "samples/synthetic_fire_grow.avi"
FPS = 10
N_FRAMES = 300          # 30 秒
GROW_K = 0.06           # 面积每秒相对增长率(6%/s → 30s 后约 5 倍)
W_OUT, H_OUT = 640, 480


def main():
    # 1) 从源视频第 0 帧,用 YOLO 精确检出火焰框并抠出来当纹理
    from ultralytics import YOLO
    cap = cv2.VideoCapture(SRC)
    ok, frame = cap.read()
    assert ok, "打不开源视频"
    model = YOLO("models/best.pt")
    r = model.predict(frame, conf=0.25, device=0, verbose=False)[0]
    fb = []
    if r.boxes is not None and len(r.boxes):
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        fb = [b for b, c in zip(xyxy, cls) if c == 1]
    assert fb, "源视频第 0 帧没检出火焰,换帧重试"
    x1, y1, x2, y2 = max(fb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # 稍微外扩一点,避免贴边裁掉火焰
    x1, y1 = max(x1 - 5, 0), max(y1 - 5, 0)
    x2, y2 = min(x2 + 5, frame.shape[1]), min(y2 + 5, frame.shape[0])
    tex = frame[y1:y2, x1:x2].copy()
    cap.release()
    print(f"[纹理] 火焰框 ({x1},{y1})-({x2},{y2}), 抠取 {tex.shape[1]}x{tex.shape[0]}")

    # 2) 计算面积线性增长曲线
    # 目标: box 面积 = A0 * (1 + GROW_K*t)。为保持长宽比,宽高各乘 sqrt。
    h0, w0 = tex.shape[:2]
    a0 = w0 * h0
    center = (W_OUT // 2, H_OUT // 2)

    # 3) 生成视频
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W_OUT, H_OUT))
    bg = np.zeros((H_OUT, W_OUT, 3), np.uint8)
    bg[:] = (30, 40, 30)  # 暗色背景

    for i in range(N_FRAMES):
        t = i / FPS
        # 面积线性增长;若超过画面容量就保持不变(封顶)
        area = a0 * (1 + GROW_K * t)
        w = int(round(w0 * np.sqrt(area / a0)))
        h = int(round(h0 * np.sqrt(area / a0)))
        w, h = max(w, 8), max(h, 8)
        w, h = min(w, W_OUT - 8), min(h, H_OUT - 8)

        resized = cv2.resize(tex, (w, h))
        x0 = center[0] - w // 2
        y0 = center[1] - h // 2
        frame_img = bg.copy()
        # 用加色混合贴上去,更接近发光火焰
        roi = frame_img[y0:y0 + h, x0:x0 + w].astype(np.int32)
        roi = np.clip(roi + resized.astype(np.int32), 0, 255).astype(np.uint8)
        frame_img[y0:y0 + h, x0:x0 + w] = roi
        writer.write(frame_img)

    writer.release()
    print(f"[完成] 已生成 {OUT}: {N_FRAMES} 帧, {N_FRAMES/FPS:.0f}s, "
          f"面积从 {a0} 线性增长约 {1 + GROW_K * N_FRAMES / FPS:.1f} 倍")


if __name__ == "__main__":
    main()
