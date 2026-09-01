"""
fire_trend.py — 火势趋势预测模块
=================================
基于连续检测帧,统计火焰"面积 / 中心位置"随时间的变化,预测:
  1) 火势趋势 : 扩大 / 稳定 / 减小
  2) 蔓延方向 : 8 方位(东 / 东南 / 南 / ...)
  3) 未来 N 秒面积变化预测(百分比)

用法(配合 ultralytics 实时检测):
    from fire_trend import FireTrendPredictor
    predictor = FireTrendPredictor()
    ...
    t = time.time() - t0
    if 检测到火:
        predictor.update(box_xyxy, frame.shape, t)
    else:
        predictor.note_no_fire(t)        # 长时间没火自动清空历史
    res = predictor.analyze(predict_seconds=5.0)
    if res:
        print(res.trend, res.direction, f"{res.pred_delta_pct:+.0f}%")

只依赖 numpy。可直接运行本文件做自测:
    python scripts/fire_trend.py
"""
import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class TrendResult:
    trend: str             # '扩大' / '稳定' / '减小'
    slope_per_sec: float   # 归一化面积变化率(1/秒)
    direction: str         # '东'/'东南'/... / '静止'
    vel_norm: tuple        # 归一化速度 (vx, vy)(1/秒),图像坐标系:+x=右 +y=下
    current_area_norm: float  # 当前平滑后的归一化面积
    predicted_area_norm: float  # 预测 N 秒后的归一化面积
    pred_delta_pct: float  # 预测 N 秒后面积变化百分比(相对当前)
    samples: int           # 参与计算的样本数
    duration: float        # 采样时间跨度(秒)


def _linreg(ts, ys):
    """一维最小二乘线性回归,返回 (斜率, 截距)。"""
    n = len(ts)
    if n < 2:
        return 0.0, float(ys[0] if len(ys) else 0.0)
    mx, my = ts.mean(), ys.mean()
    denom = ((ts - mx) ** 2).sum()
    if abs(denom) < 1e-12:
        return 0.0, my
    slope = ((ts - mx) * (ys - my)).sum() / denom
    return float(slope), float(my - slope * mx)


def _angle_to_dir(vx, vy):
    """把速度向量映射到 8 方位。图像坐标:+x=右(东), +y=下(南)。"""
    if abs(vx) < 1e-4 and abs(vy) < 1e-4:
        return '静止'
    ang = (np.degrees(np.arctan2(vy, vx)) + 360.0) % 360.0
    dirs = ['东', '东南', '南', '西南', '西', '西北', '北', '东北']
    idx = int((ang + 22.5) // 45) % 8
    return dirs[idx]


def union_box(boxes_xyxy):
    """把所有火焰框合并成一个并集外接框 [x1,y1,x2,y2]。

    画面里通常有多团火/多个检测框,只取最大的一只会让信号在框与框之间跳
    (中心乱跳、面积抖动)。合并成并集后,代表"整片火"的信号稳定得多。
    """
    boxes = list(boxes_xyxy)
    if len(boxes) == 1:
        return np.asarray(boxes[0], dtype=float)
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return np.array([x1, y1, x2, y2])


class FireTrendPredictor:
    def __init__(self, max_samples=90, min_samples=6, min_duration=2.0,
                 trend_threshold_rel=0.02, stale_seconds=3.0, ema_alpha=0.35):
        self.max_samples = max_samples
        self.min_samples = min_samples            # 至少多少个样本才开始算
        self.min_duration = min_duration          # 至少跨多少秒才开始算
        self.trend_threshold_rel = trend_threshold_rel  # 面积变化率阈值(相对当前面积/秒)
        self.stale_seconds = stale_seconds        # 超过这么久没火,清空历史
        self.ema_alpha = ema_alpha                # 指数平滑系数(0=完全不动,1=不平滑)
        self.history = deque(maxlen=max_samples)  # (t, ema_area, ema_cx, ema_cy)
        self._ema = None                          # (area, cx, cy)

    def reset(self):
        self.history.clear()
        self._ema = None

    def update(self, box_xyxy, frame_shape, t):
        """每帧喂一次火焰的检测框(取最大的一只即可)。

        先做 EMA 指数平滑再入历史:火焰 bbox 天然抖动(帧间常跳 10~20%),
        平滑后拟合出来的趋势线更稳,预测不再被单帧抖动带偏。
        """
        x1, y1, x2, y2 = box_xyxy
        H, W = frame_shape[0], frame_shape[1]
        area = max(float((x2 - x1) * (y2 - y1)), 0.0)
        area_norm = area / float(W * H)
        cx_norm = (x1 + x2) / 2.0 / W
        cy_norm = (y1 + y2) / 2.0 / H

        # 距上一个样本太久说明中间断过,历史已过时,清掉重新积累
        if self.history and t - self.history[-1][0] > self.stale_seconds:
            self.history.clear()
            self._ema = None

        a = self.ema_alpha
        if self._ema is None:
            self._ema = (area_norm, cx_norm, cy_norm)
        else:
            ea, ex, ey = self._ema
            self._ema = (a * area_norm + (1 - a) * ea,
                         a * cx_norm + (1 - a) * ex,
                         a * cy_norm + (1 - a) * ey)
        ea, ex, ey = self._ema
        self.history.append((t, ea, ex, ey))

    def note_no_fire(self, t):
        """这一帧没检测到火。太久没火就清空历史,避免拿旧数据预测。"""
        if self.history and t - self.history[-1][0] > self.stale_seconds:
            self.history.clear()

    def analyze(self, predict_seconds=5.0):
        """返回 TrendResult;样本不足 / 跨度太短返回 None。"""
        n = len(self.history)
        if n < self.min_samples:
            return None
        ts = np.array([h[0] for h in self.history])
        if ts[-1] - ts[0] < self.min_duration:
            return None
        areas = np.array([h[1] for h in self.history])
        cxs = np.array([h[2] for h in self.history])
        cys = np.array([h[3] for h in self.history])

        slope_a, _ = _linreg(ts, areas)   # 面积变化率
        vx, _ = _linreg(ts, cxs)          # 中心横向速度
        vy, _ = _linreg(ts, cys)          # 中心纵向速度

        cur_area = areas[-1]
        rel = slope_a / cur_area if cur_area > 1e-6 else 0.0
        if rel > self.trend_threshold_rel:
            trend = '扩大'
        elif rel < -self.trend_threshold_rel:
            trend = '减小'
        else:
            trend = '稳定'

        pred_delta_pct = rel * predict_seconds * 100.0
        return TrendResult(
            trend=trend,
            slope_per_sec=float(slope_a),
            direction=_angle_to_dir(float(vx), float(vy)),
            vel_norm=(float(vx), float(vy)),
            current_area_norm=float(cur_area),
            predicted_area_norm=float(cur_area + slope_a * predict_seconds),
            pred_delta_pct=float(pred_delta_pct),
            samples=n,
            duration=float(ts[-1] - ts[0]),
        )


# --------------------------------------------------------------------------
# 自测:不依赖无人机,用合成数据验证判断正确
# --------------------------------------------------------------------------
if __name__ == '__main__':
    def run_case(name, make_box):
        p = FireTrendPredictor()
        for i in range(60):
            p.update(make_box(i), (720, 1280), i * 0.1)   # 每秒 10 帧,共 6 秒
        res = p.analyze(predict_seconds=5.0)
        print(f"[{name}] -> trend={res.trend}, dir={res.direction}, "
              f"pred_delta={res.pred_delta_pct:+.0f}%, samples={res.samples}")
        return res

    # 1) 火势扩大:框越变越大,中心向右上移动 → 扩大 + 东北
    def grow(i):
        w = 20 * (1 + i * 0.05)   # 20 → 79
        return (400 + i, 300 - i, 400 + i + w, 300 - i + w)
    r1 = run_case('扩大', grow)

    # 2) 火势减小:框越变越小,中心不动 → 减小
    def shrink(i):
        w = 200 - i * 2.3         # 200 → 64
        return (600, 300, 600 + w, 300 + w)
    r2 = run_case('减小', shrink)

    # 3) 稳定:框大小不变 → 稳定 + 静止
    def steady(i):
        return (600, 300, 800, 500)
    r3 = run_case('稳定', steady)

    # 4) 数据不足:只给 2 帧 → 应返回 None
    p4 = FireTrendPredictor()
    for i in range(2):
        p4.update((600, 300, 800, 500), (720, 1280), i * 0.1)
    r4 = p4.analyze()

    assert r1.trend == '扩大' and r1.direction == '东北', r1
    assert r2.trend == '减小', r2
    assert r3.trend == '稳定' and r3.direction == '静止', r3
    assert r4 is None
    print('\n[OK] 自测通过:趋势判断、方向映射、数据不足保护全部正确')
