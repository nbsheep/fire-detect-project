from ultralytics import YOLO

# 用官方预训练的 yol11n(nano,最小最快),在它基础上微调 —— 这叫迁移学习,
# 比从零训练快得多、效果好得多。第一次运行会自动下载 yolo11n.pt。
model = YOLO("yolo11n.pt")

if __name__ == "__main__":
    model.train(
        data="data/dfire.yaml",   # 上一步写的配置
        epochs=50,                 # 训练轮数,先 50,不够再加
        imgsz=640,                 # 输入图片尺寸,640 是标准值
        batch=16,                  # 一批多少张,16 对 16G 显存很稳;爆显存就调小到 8
        device=0,                  # 用第 0 块 GPU;写 "cpu" 则用 CPU(很慢)
        workers=4,                 # 数据加载线程,Windows 上别调太高
        project="runs",            # 输出目录
        name="fire_yolo11n",       # 本次训练的名字
        patience=15,               # 15 轮没提升就早停,省时间
    )