import time
import cv2
from ultralytics import YOLO
from reporter import InspectionReporter
from alarm import AlarmDebouncer

# ==================== 配置区 ====================
WEIGHTS = "models/best.pt"
SOURCE  = "samples/test.mp4"   # 测试用视频(无摄像头); 0=摄像头; "rtmp://..."/"rtsp://..."=无人机图传
CONF    = 0.35         # 置信度阈值
TRIGGER = 5            # 连续几帧检到才报警
RELEASE = 15           # 连续几帧没检到才解除
SHOW    = True         # 是否弹窗显示
# ================================================

def get_gps():
    """占位:接飞控后在这里返回 (lat, lon)。暂时返回 None。"""
    return None

def open_stream(src):
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap if cap.isOpened() else None

def main():
    model    = YOLO(WEIGHTS)
    reporter = InspectionReporter()
    # 火和烟各用一个去抖器,分别判断
    alarms   = {"fire": AlarmDebouncer(TRIGGER, RELEASE),
                "smoke": AlarmDebouncer(TRIGGER, RELEASE)}

    cap = open_stream(SOURCE)
    if cap is None:
        print(f"[错误] 打不开视频源:{SOURCE}")
        return
    print("[巡检] 开始。按 q 退出(窗口激活时)。")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # 视频文件放完了就结束;实时流则尝试重连
                if isinstance(SOURCE, str) and SOURCE.startswith(("rtsp", "rtmp")):
                    print("[警告] 读帧失败,重连中")
                    cap.release(); time.sleep(2)
                    cap = open_stream(SOURCE)
                    if cap is None:
                        time.sleep(2)
                    continue
                break

            results   = model.predict(frame, conf=CONF, device=0, verbose=False)
            r         = results[0]
            annotated = r.plot()

            # 统计这一帧每类的最高置信度
            per_class = {"fire": 0.0, "smoke": 0.0}
            for box in r.boxes:
                name = model.names[int(box.cls[0])]
                if name in per_class:
                    per_class[name] = max(per_class[name], float(box.conf[0]))

            # 分别更新去抖器;只在"刚触发"时记事件
            for label, deb in alarms.items():
                detected = per_class[label] > 0
                state = deb.update(detected)
                if state == "RAISE":
                    print(f"🔥 [报警] 确认 {label}!conf={per_class[label]:.2f}")
                    reporter.log_event(annotated, label, per_class[label], gps=get_gps())
                elif state == "CLEAR":
                    print(f"✔ [解除] {label} 报警解除")

            if SHOW:
                cv2.imshow("Patrol - press q to quit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\n[巡检] 手动中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        reporter.save_report()   # 无论如何都出报告

if __name__ == "__main__":
    main()