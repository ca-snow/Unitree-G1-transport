"""
影子模式视觉报告:把 m4_shadow_*.csv 汇总成"纯视觉"的决策证据表。
输出:逐类别统计"假如上纯视觉"的定位链误差(估计值 vs 真值,在M4循环 A 桌再定位停靠点实测),并对照BC停靠±3cm训练包线给出判定。
"""

import csv
import glob
import math
import os
import statistics as st
import sys

# BC 策略的停靠训练包线(±cm)
BC_ENVELOPE_CM = 3.0
# 脚本化链路自身的停靠散布(M4批量实验第1~5轮、通过判定的停靠标准差)视觉误差必须在包线里给它留出这部分空间
SCRIPT_DOCK_SD_CM = 1.2
# 估计值离真值超过这个距离,说明单像素深度采样完全没打到工件(打到了桌面/地面/背景)——这类样本按"深度脱靶"单独计数,不混进误差统计
DEPTH_MISS_CM = 15.0


def load(paths):
    #读取csv所有影子数据,合并成字典行列表。
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def fnum(r, k):
    #安全取数:字段缺失或非数值时返回 None。
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return None


def stat(vals):
    #基本统计:返回(均值, 标准差, 最小值, 最大值, 样本数);空则 None。
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    mu = st.mean(vals)
    sd = st.pstdev(vals) if len(vals) > 1 else 0.0
    return mu, sd, min(vals), max(vals), len(vals)


def fmt(s, signed=True):
    #把 stat()结果排版成 "均值 ± 标准差 [最小, 最大]" 的表格单元格。
    if s is None:
        return "-"
    mu, sd, lo, hi, n = s
    sign = "+" if signed else ""
    return f"{mu:{sign}.2f} ± {sd:.2f} [{lo:{sign}.1f}, {hi:{sign}.1f}]"


def main():
    paths = []
    for a in sys.argv[1:]:
        paths.extend(glob.glob(a))
    if not paths:
        paths = sorted(glob.glob("m4_shadow_*.csv"))
    if not paths:
        print("no shadow csv found (pass paths or run next to m4_shadow_*.csv)")
        return
    rows = load(paths)
    print(f"loaded {len(rows)} shadow measurements from {len(paths)} file(s)\n")

    lines = []
    lines.append("## 全视觉停靠方案影子验证（M4 循环实测，未参与控制）")
    lines.append("")
    lines.append(f"- 测量点：A 桌前 2.5 m 重整位，D435 规格头部相机（1280x720，HFOV 87°），低头 15°")
    lines.append(f"- 视觉链路：像素中心 + 深度中值 → 反投影 → 类别修正（标定常数）→ 工件根坐标估计")
    lines.append(f"- 判据：BC 停靠训练包线 ±{BC_ENVELOPE_CM:.0f} cm；脚本链路自身停靠散布 "
                 f"~±{SCRIPT_DOCK_SD_CM:.1f} cm，视觉误差需在剩余预算内")
    lines.append("")
    lines.append("| 类别 | n | 深度脱靶率 | 前后误差 (cm) | 左右误差 (cm) | 高度误差 (cm) | 平面误差 |xy| (cm) | 3σ 平面误差 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    overall_bad = []
    miss_note = []
    for cls in ("cube", "bolt", "nut", "drill"):
        sel = [r for r in rows if r["cls"] == cls]
        if not sel:
            continue
        good = [r for r in sel
                if (fnum(r, "err_xy_cm") is not None
                    and abs(fnum(r, "err_xy_cm")) <= DEPTH_MISS_CM)]
        n_miss = len(sel) - len(good)
        miss_pct = 100.0 * n_miss / len(sel)
        if miss_pct >= 10.0:
            miss_note.append(f"{cls} {n_miss}/{len(sel)}")
        s_f = stat([fnum(r, "err_fwd_cm") for r in good])
        s_l = stat([fnum(r, "err_left_cm") for r in good])
        s_z = stat([fnum(r, "err_z_cm") for r in good])
        s_xy = stat([fnum(r, "err_xy_cm") for r in good])
        if s_xy is None:
            lines.append(f"| {cls} | {len(sel)} | {n_miss}/{len(sel)} ({miss_pct:.0f}%) "
                         f"| - | - | - | - | - | 深度不可用 |")
            overall_bad.append(cls)
            continue
        # 最坏方向的 3σ 上界:|系统偏差| + 3 倍平面误差标准差
        three_sig = abs(s_xy[0]) + 3.0 * s_xy[1]
        budget = math.sqrt(max(BC_ENVELOPE_CM ** 2 - SCRIPT_DOCK_SD_CM ** 2, 0.0))
        if miss_pct >= 10.0:
            verdict = f"深度脱靶 {miss_pct:.0f}%，链路不可用"
        elif three_sig <= budget:
            verdict = "在预算内"
        else:
            verdict = f"超预算 ({three_sig:.1f} > {budget:.1f})"
        if three_sig > budget or miss_pct >= 10.0:
            overall_bad.append(cls)
        lines.append(f"| {cls} | {len(sel)} | {n_miss}/{len(sel)} ({miss_pct:.0f}%) "
                     f"| {fmt(s_f)} | {fmt(s_l)} | {fmt(s_z)} "
                     f"| {fmt(s_xy, signed=False)} | {three_sig:.1f} cm | {verdict} |")
    lines.append("")
    lines.append("### 结论要点（结合 head_cam_calib 2026-08-14 标定）")
    lines.append("")
    pts = []
    pts.append("停靠点处工件在头部相机 FOV 之外（需低头 ≥45°，真机 G1 头部相机俯仰固定），"
               "全视觉方案只能在 2.5 m 站位测量后航位推算，误差随行走继续累积；")
    pts.append("深度测的是朝向相机的表面而非工件根坐标，修正常数随视角变号"
               "（drill 在停靠点 +3.8 cm、2.5 m 处 -5.4 cm），无法用单一常数覆盖全程；")
    if miss_note:
        pts.append("单像素深度采样在细长/薄壁工件上大量脱靶"
                   "（打穿工件命中背景，估计偏差以米计）：脱靶 "
                   + "、".join(miss_note)
                   + "——鲁棒化需要邻域中值/分割掩码，链路复杂度显著上升；")
    pts.append("nut 资产的 USD 原点位于可见网格下方 ~7.5 cm，视觉检测点与抓取参考点"
               "系统性偏离，需逐资产标定；")
    pts.append("上表为实测的全视觉估计误差；叠加真实检测器的像素误差"
               "（2.5 m 处 5.4 mm/像素，YOLO 中心点 ±3 像素即 ±1.6 cm）后，"
               + ("以下类别超出 ±3 cm 包线预算或链路不可用：" + ", ".join(overall_bad)
                  if overall_bad else "余量已非常有限") + "；")
    pts.append("结论：视觉降级为类别 gate（YOLO 只判类别、匹配任务调度），"
               "定位保持已验证的脚本链路。")
    lines.extend(f"{i}. {p}" for i, p in enumerate(pts, 1))
    text = "\n".join(lines)
    print(text)
    out = os.path.join(os.path.dirname(os.path.abspath(paths[0])),
                       "vision_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n[report] written -> {out}")


if __name__ == "__main__":
    main()
