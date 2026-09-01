import time
import numpy as np
import cv2
from ultralytics import YOLO
from fire_trend import FireTrendPredictor, union_box

WEIGHTS  = "models/best.pt"
# 调试用 0 走本机摄像头;接无人机时用 MediaMTX 提供的地址(见桌面 RTMP 指南):
#   rtmp://127.0.0.1:1935/live/drone  或  rtsp://127.0.0.1:8554/live/drone
RTSP_URL = "rtsp://127.0.0.1:8554/live/drone"
CONF     = 0.35                              # 置信度阈值

FIRE_CLASS     = 1       # dfire.yaml: 0=smoke, 1=fire
PREDICT_SECONDS = 5.0    # 预测未来几秒的面积变化


def open_stream(url):
    """打开视频流,失败返回 None。"""
    cap = cv2.VideoCapture(url)
    # 减小缓冲,降低延迟(部分后端支持)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap if cap.isOpened() else None


# --------------------------------------------------------------------------
# 中文叠加绘制:有中文字体画中文,没有则退回英文(避免乱码/问号)
# --------------------------------------------------------------------------
def _get_cn_font(size):
    try:
        from PIL import ImageFont
        for p in (r"C:/Windows/Fonts/msyh.ttc",
                  r"C:/Windows/Fonts/simhei.ttf",
                  r"C:/Windows/Fonts/simsun.ttc"):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    except Exception:
        pass
    return None


def draw_cn(img, text, pos, size=24, color=(0, 255, 255)):
    """在图像上画文字,优先中文,失败退回 ASCII。"""
    try:
        from PIL import Image, ImageDraw
        font = _get_cn_font(size)
        if font is not None:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ImageDraw.Draw(pil).text(pos, text, font=font, fill=color[::-1])
            img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            return
    except Exception:
        pass
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                size / 32.0, color, 2)


def main():
    model = YOLO(WEIGHTS)
    predictor = FireTrendPredictor()
    t0 = time.time()
    cap = None
    while True:
        # 断线重连
        if cap is None or not cap.isOpened():
            print("[连接] 正在连接视频流 ...")
            cap = open_stream(RTSP_URL)
            if cap is None:
                print("[连接] 失败,3 秒后重试")
                time.sleep(3)
                continue
            print("[连接] 成功")

        ok, frame = cap.read()
        if not ok:
            print("[警告] 读帧失败,尝试重连")
            cap.release()
            cap = None
            continue

        # 推理(单帧)。verbose=False 不刷屏
        results = model.predict(frame, conf=CONF, device=0, verbose=False)
        annotated = results[0].plot()   # 画好框的图(numpy 数组)

        # ---- 火势趋势预测 ----
        t = time.time() - t0
        boxes = results[0].boxes
        fire_boxes = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for b, c in zip(xyxy, cls):
                if c == FIRE_CLASS:
                    fire_boxes.append(b)

        if fire_boxes:
            # 用并集外接框代表"整片火":多火团时比单只框稳定,不会乱跳
            predictor.update(union_box(fire_boxes), frame.shape, t)
        else:
            predictor.note_no_fire(t)

        res = predictor.analyze(predict_seconds=PREDICT_SECONDS)
        if res is not None:
            line1 = f"火势趋势: {res.trend} | 蔓延方向: {res.direction}"
            line2 = (f"预计 {PREDICT_SECONDS:.0f}s 后面积 "
                     f"{res.pred_delta_pct:+.0f}% | 样本 {res.samples}个/{res.duration:.1f}s")
            print(f"[{time.strftime('%H:%M:%S')}] {line1} | {line2}")
            draw_cn(annotated, line1, (10, 10), 28, (0, 255, 255))
            draw_cn(annotated, line2, (10, 46), 24, (0, 255, 255))
        else:
            draw_cn(annotated, "火势趋势采样中...", (10, 10), 24, (180, 180, 180))

        cv2.imshow("Fire/Smoke Detection + Trend - press q to quit", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
