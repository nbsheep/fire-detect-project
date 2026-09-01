"""
fire_trend_verify.py — 火势趋势预测"回测验证"
=============================================
拿一个真实视频,逐帧跑 YOLO 检测 → 喂给 FireTrendPredictor。
在每个做出预测的时刻 t,记下"预测的 P 秒后面积";真的处理到 t+P 时,
拿当时的实际面积对比,统计:

  1. 面积预测误差:平均相对误差、中位误差、±20% / ±50% 命中率
  2. 趋势方向预测:预测"扩大/减小/稳定"和实际走势是否一致(准确率)
  3. 一张对比图:面积时间曲线 + 预测点(绿=命中,红=偏差大) + 实际到达点
     以及误差分布直方图

用法:
  python scripts/fire_trend_verify.py <视频路径> [预测秒数]

说明:
  - 视频是"固定镜头 + 火势自然变化"时预测最准;镜头移动/缩放会拉高误差,
    这也正是理论部分的局限,误差大时注意看是不是镜头在动。
  - 每帧时间用 帧序号/帧率 计算,掉帧/卡顿不影响时间轴。
"""
import sys
import bisect
import numpy as np
import cv2
from ultralytics import YOLO
from fire_trend import FireTrendPredictor, union_box

WEIGHTS = "models/best.pt"
FIRE_CLASS = 1
CONF = 0.25
HORIZON = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0   # 预测未来几秒


def pick_fire_boxes(results):
    """从一帧结果里挑出所有 fire 类检测框。"""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    out = []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for b, c in zip(xyxy, cls):
        if c == FIRE_CLASS:
            out.append(b)
    return out


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 打不开视频: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[视频] {video_path}\n      分辨率 {int(cap.get(3))}x{int(cap.get(4))}, "
          f"fps={fps:.1f}, 共 {total} 帧, 预测窗口 {HORIZON:.0f}s")

    model = YOLO(WEIGHTS)
    predictor = FireTrendPredictor()

    timeline = []      # 每帧: (t, actual_area_norm) 供回测查真值
    predictions = []   # 每条: (t_pred, predicted_area_norm, current_area, trend)
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_no / fps

        results = model.predict(frame, conf=CONF, device=0, verbose=False)
        fb = pick_fire_boxes(results)

        if fb:
            box = union_box(fb)
            x1, y1, x2, y2 = box
            area_norm = max(float((x2 - x1) * (y2 - y1)), 0.0) / (frame.shape[0] * frame.shape[1])
            predictor.update(box, frame.shape, t)
            timeline.append((t, area_norm))

            res = predictor.analyze(predict_seconds=HORIZON)
            if res is not None:
                # 预测值用预测器内部(EMA 平滑后)算好的面积,与显示层保持一致
                predictions.append((t, res.predicted_area_norm, res.current_area_norm, res.trend))
        else:
            predictor.note_no_fire(t)

        frame_no += 1
        if frame_no % 100 == 0:
            print(f"  已处理 {frame_no}/{total} 帧")

    cap.release()

    if not timeline:
        print("[结果] 整个视频没检测到火焰,无法验证。换个有明火的视频试试。")
        return
    if not predictions:
        print("[结果] 检测到了火焰,但样本/时长不足以产生预测(火焰持续太短)。")
        return

    # ---- 回测:对每条预测,查真实到达 t+HORIZON 时的实际面积 ----
    ts = np.array([x[0] for x in timeline])
    areas = np.array([x[1] for x in timeline])
    rows = []
    for t_pred, pred_area, cur_area, trend in predictions:
        t_fut = t_pred + HORIZON
        i = bisect.bisect_left(ts, t_fut)
        if i >= len(ts):
            continue  # 视频在到达预测时刻前就结束了,不算
        actual_strict = areas[i]                                    # 单帧瞬时值(受火焰抖动影响大)
        # 中位真值:取到达时刻前后 0.5s 窗口的中位面积,过滤掉单帧抖动
        lo = bisect.bisect_left(ts, t_fut - 0.5)
        hi = bisect.bisect_right(ts, t_fut + 0.5)
        actual_median = float(np.median(areas[max(lo, i - 30):min(hi, i + 30)]))
        rel_err = (pred_area - actual_strict) / max(actual_strict, 1e-9)
        rel_err_med = (pred_area - actual_median) / max(actual_median, 1e-9)
        # 实际走势:未来中位面积 相对 当前面积
        actual_trend = ('扩大' if actual_median > cur_area * 1.05
                        else ('减小' if actual_median < cur_area * 0.95 else '稳定'))
        rows.append((t_pred, pred_area, actual_median, rel_err, rel_err_med, trend, actual_trend))

    if not rows:
        print("[结果] 视频太短,没有一条预测能等到 P 秒后的真值。换长一点的视频(建议 >15s)。")
        return

    rel_errs = np.array([r[3] for r in rows])          # 对单帧瞬时值
    rel_meds = np.array([r[4] for r in rows])          # 对局部中位值(推荐看这个)
    trend_ok = np.mean([1.0 if r[5] == r[6] else 0.0 for r in rows]) * 100

    def line(name, e):
        return (f"{name:<18}: 平均 {np.mean(np.abs(e))*100:.0f}% | "
                f"中位 {np.median(np.abs(e))*100:.0f}% | "
                f"≤20% {np.mean(np.abs(e)<=0.2)*100:.0f}% | "
                f"≤50% {np.mean(np.abs(e)<=0.5)*100:.0f}%")

    print("\n============== 验证结果 ==============")
    print(f"可评估的预测数      : {len(rows)} 条 (覆盖 {rows[-1][0]:.1f}s 之前)")
    print(f"{'误差vs单帧瞬时值':<18}平均{np.mean(np.abs(rel_errs))*100:.0f}% 中位{np.median(np.abs(rel_errs))*100:.0f}% (最严苛)")
    print(f"{'误差vs局部中位值':<18}平均{np.mean(np.abs(rel_meds))*100:.0f}% 中位{np.median(np.abs(rel_meds))*100:.0f}% (推荐,过滤抖动)")
    print(f"趋势方向判断准确率  : {trend_ok:.0f}%")
    print("======================================")
    print("注:若误差普遍大,先检查视频镜头是不是在移动/缩放(那会破坏面积预测)。")

    _plot(ts, areas, rows)
    return rows


def _plot(ts, areas, rows):
    """输出对比图:面积曲线 + 预测点(绿=命中≤20%,红=偏差大) + 误差直方图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for f in (r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/simhei.ttf"):
            try:
                font_manager.fontManager.addfont(f)
                plt.rcParams["font.sans-serif"] = [font_manager.FontProperties(fname=f).get_name()]
                break
            except Exception:
                continue
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(ts, areas * 100, lw=1.5, label="实际火焰面积(占画面%)")
    for t_pred, pred, actual, rel, *_ in rows:
        ok = abs(rel) <= 0.20
        ax.scatter(t_pred, pred * 100, color="lime" if ok else "red", s=28, zorder=3)
        ax.plot([t_pred, t_pred + HORIZON], [pred * 100, actual * 100],
                color="lime" if ok else "red", lw=0.8, alpha=0.6)
    ax.set_title(f"火势趋势回测 — 绿=预测误差≤20%, 红=偏差大(水平线=预测→实际)")
    ax.set_xlabel("时间(秒)")
    ax.set_ylabel("火焰面积(占画面%)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    rels = np.array([abs(r[3]) for r in rows])
    ax.hist(rels * 100, bins=20, color="steelblue", alpha=0.8)
    ax.axvline(20, color="lime", ls="--", lw=1.5, label="±20% 线")
    ax.axvline(50, color="orange", ls="--", lw=1.5, label="±50% 线")
    ax.set_title("预测误差分布(绝对相对误差%)")
    ax.set_xlabel("误差%")
    ax.set_ylabel("预测条数")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = "runs/verify/fire_trend_verify.png"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n[图] 对比图已保存: {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
