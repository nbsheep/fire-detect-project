"""train_status.py — 查看训练进度
用法: python scripts/train_status.py
"""
import csv
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RUN = Path(__file__).resolve().parent.parent / "runs" / "detect" / "runs" / "fire_yolo26s_plus"


def main():
    f = RUN / "results.csv"
    if not f.exists():
        print("还没有训练记录")
        return
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    last = rows[-1]
    ep = int(last["epoch"])
    total = 80
    elapsed = float(last["time"])
    # 续训会重置时间列,取相邻轮时间差的正值来估算每轮耗时
    times = [float(r["time"]) for r in rows]
    deltas = [b - a for a, b in zip(times, times[1:]) if 0 < b - a < 3600]
    avg = deltas[-1] if deltas else elapsed / max(ep, 1)
    eta_min = (total - ep) * avg / 60
    print(f"进度: {ep}/{total} 轮 | 已用 {elapsed/3600:.1f}h | 预计还需 ~{eta_min:.0f} 分钟")
    print(f"mAP50={float(last['metrics/mAP50(B)']):.3f}  "
          f"mAP50-95={float(last['metrics/mAP50-95(B)']):.3f}  "
          f"P={float(last['metrics/precision(B)']):.3f}  R={float(last['metrics/recall(B)']):.3f}")
    print("\n最近 8 轮 mAP50 走势:")
    for r in rows[-8:]:
        bar = "#" * int(float(r["metrics/mAP50(B)"]) * 40)
        print(f"  轮{int(r['epoch']):>3}  {float(r['metrics/mAP50(B)']):.3f} {bar}")
    if ep >= total:
        print("\n[训练已完成] 最终权重: " + str(RUN / "weights" / "best.pt"))
    else:
        print(f"\n(每轮约 {avg/60:.1f} 分钟; 实时日志: runs/train_yolo26s_resume.log)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()
