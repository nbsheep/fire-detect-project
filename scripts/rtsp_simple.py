from ultralytics import YOLO

# 换成你的权重路径
model = YOLO("models/best.pt")

# 调试用 0 走本机摄像头;接无人机时用 MediaMTX 提供的地址(见桌面 RTMP 指南):
#   rtmp://127.0.0.1:1935/live/drone  或  rtsp://127.0.0.1:8554/live/drone
RTSP_URL = 0

if __name__ == "__main__":
    # stream=True 用生成器逐帧处理,省内存;show=True 弹窗实时显示
    model.predict(source=RTSP_URL, show=True, conf=0.25, device=0, stream=True)
    # 注意:stream=True 时 predict 返回生成器,需要遍历它才会真正开始跑,见 1.2