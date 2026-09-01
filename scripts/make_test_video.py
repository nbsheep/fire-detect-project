"""
用 D-Fire 测试集里【带火/烟标注】的图片,合成一段测试视频 samples/test.mp4。
没有摄像头/无人机时,用它来验证 patrol.py、report_demo.py 整条链路。
"""
import os
import cv2

LABELS = "data/dfire/test/labels"
IMAGES = "data/dfire/test/images"
OUT    = "samples/test.mp4"
SIZE   = (640, 640)   # 统一尺寸
N      = 120          # 取多少张有标注的图
FRAMES_PER_IMG = 4    # 每张图持续几帧(越大播得越慢)
FPS    = 24

def main():
    # 找出有火/烟标注(非空)的图片
    picked = []
    for txt in sorted(os.listdir(LABELS)):
        if not txt.endswith(".txt"):
            continue
        p = os.path.join(LABELS, txt)
        if os.path.getsize(p) == 0:      # 空标注 = 无火无烟,跳过
            continue
        img = os.path.join(IMAGES, txt.replace(".txt", ".jpg"))
        if os.path.exists(img):
            picked.append(img)
        if len(picked) >= N:
            break

    if not picked:
        print("[错误] 没找到带标注的图片,检查 data/dfire/test 路径")
        return

    os.makedirs("samples", exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for img_path in picked:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, SIZE)
        for _ in range(FRAMES_PER_IMG):
            writer.write(img)
    writer.release()
    print(f"[完成] 已生成 {OUT},用了 {len(picked)} 张图,"
          f"约 {len(picked)*FRAMES_PER_IMG/FPS:.0f} 秒")

if __name__ == "__main__":
    main()
