# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
对 transport 采集的 BC 检查点做离线闭环诊断（不需要 Isaac）。
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

ROWS = 415                      # 每条 episode 的行数（830 个 BC tick / 每 2 tick 存 1 行）

# 110 维观测布局（镜像 transport_demo 的 BC 观测拼装顺序）
BLOCKS = {
    "body_qpos":   slice(0, 29),     # 29 个身体关节角
    "body_qvel":   slice(29, 58),    # 29 个身体关节角速度
    "hand_qpos":   slice(58, 72),    # 14 个手指关节角
    "obj_rel_L":   slice(72, 75),    # 物体相对左腕位置
    "obj_rel_R":   slice(75, 78),    # 物体相对右腕位置（抓取的关键信号）
    "last_action": slice(78, 106),   # 上一步动作（捷径嫌疑通道）
    "phase":       slice(106, 107),  # episode 相位
    "place_rel":   slice(107, 110),  # 放置目标相对位置
}
LAST_ACTION = BLOCKS["last_action"]

# 专家各相位的边界（单位：BC tick；transport "抬起式起始" 的 830 tick 周期）
PHASES = [
    ("HOLD", 0, 19), ("TRAV", 20, 104), ("FLIP", 105, 194),
    ("DESCEND", 195, 274), ("INSERT", 275, 334), ("CLOSE", 335, 424),
    ("LIFT", 425, 504), ("CARRY", 505, 609), ("LOWER", 610, 689),
    ("RELEASE", 690, 739), ("SIDE", 740, 764), ("RETREAT", 765, 829),
]

class SharedPolicy(nn.Module):
    #bc_train.py 中模型的镜像定义，保证 state_dict 能 1:1 加载。

    #结构：共享 [256,128,128] ELU 主干 + 策略头 + 价值头 + log_std 参数。
    #输入 obs [batch, 110]，输出确定性动作 [batch, 28]。
    

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        self.policy_layer = nn.Linear(128, act_dim)
        self.value_layer = nn.Linear(128, 1)
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        #前向：obs [batch,110] -> 动作 [batch,28]（只走策略头）。
        return self.policy_layer(self.net_container(obs))


def phase_rows(lo_tick: int, hi_tick: int) -> slice:
    #把相位的 tick 区间 [lo, hi] 换算成数据行的切片。

    #录制的行落在偶数 tick 0,2,...,828 上，所以 行号 = tick // 2。
    
    return slice(lo_tick // 2, hi_tick // 2 + 1)


def main():
    #诊断入口：加载数据+检查点，依次跑教师强制、free-run、置换重要性三个测试。
    parser = argparse.ArgumentParser(description="Offline BC shortcut / closed-loop diagnosis.")
   
    parser.add_argument("--dataset", type=str, required=True)
    # --checkpoint：bc_train.py 输出的 .pt 检查点
    parser.add_argument("--checkpoint", type=str, required=True)
    # --perm_rows：置换重要性测试的采样行数上限（10 万行足够统计稳定，又不至于占满显存）
    parser.add_argument("--perm_rows", type=int, default=100000)
    # --device / --seed：常规设备选择与随机种子
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 加载数据集并整形成 [episode, 行, 维] 的三维视图 ----
    obs_list, act_list = [], []
    for path in args.dataset.split(","):
        d = np.load(path.strip())
        obs_list.append(d["obs"])
        act_list.append(d["actions"])
        print(f"[diag] loaded {d['obs'].shape[0]} rows from {path.strip()}")
    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    # 行数必须是 415 的整数倍，否则不是完整的 transport episode 集合
    assert obs.shape[0] % ROWS == 0, f"{obs.shape[0]} rows is not a multiple of {ROWS}"
    n_ep = obs.shape[0] // ROWS
    obs3 = torch.as_tensor(obs.reshape(n_ep, ROWS, -1), dtype=torch.float32)  # [n_ep, 415, 110]
    act3 = torch.as_tensor(act.reshape(n_ep, ROWS, -1), dtype=torch.float32)  # [n_ep, 415, 28]
    print(f"[diag] {n_ep} episodes x {ROWS} rows | obs_dim={obs3.shape[-1]} act_dim={act3.shape[-1]}")

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[diag] checkpoint {args.checkpoint} (trained val MSE {ckpt.get('val_mse', float('nan')):.6f})")

    # ---- 测试 1+2：教师强制 vs last_action 自由运行，沿 episode 时间轴
    # 同步推进（batch 维 = 全部 episode 一起算）----
    tf_err = torch.zeros(n_ep, ROWS)     # 每行的 MSE，教师强制（观测全用真值）
    fr_err = torch.zeros(n_ep, ROWS)     # 每行的 MSE，last_action 自由运行
    prev_pred = None
    with torch.no_grad():
        for tr in range(ROWS):
            o_t = obs3[:, tr, :].to(args.device)   # 第 tr 行的真值观测 [n_ep,110]
            a_t = act3[:, tr, :].to(args.device)   # 第 tr 行的专家动作 [n_ep,28]
            # 测试 1：教师强制——观测原封不动喂网络
            p_tf = model(o_t)
            tf_err[:, tr] = ((p_tf - a_t) ** 2).mean(dim=-1).cpu()
            # 测试 2：free-run——仅把 last_action 块换成网络自己上一步的输出，
            # 其余观测字段仍是真值（隔离捷径反馈回路，其他因素不变）
            o_fr = o_t.clone()
            if prev_pred is not None:
                o_fr[:, LAST_ACTION] = prev_pred
            p_fr = model(o_fr)
            fr_err[:, tr] = ((p_fr - a_t) ** 2).mean(dim=-1).cpu()
            prev_pred = p_fr             # 关键：网络"吃"自己的输出，误差会随时间复利

    print("\n[diag] per-phase MSE: teacher forcing vs last-action free-run")
    print(f"  {'phase':>8} {'teacher':>10} {'free-run':>10} {'ratio':>7}")
    for name, lo, hi in PHASES:
        rs = phase_rows(lo, hi)
        tf_m = tf_err[:, rs].mean().item()
        fr_m = fr_err[:, rs].mean().item()
        print(f"  {name:>8} {tf_m:10.6f} {fr_m:10.6f} {fr_m / max(tf_m, 1e-12):7.1f}x")
    tf_all, fr_all = tf_err.mean().item(), fr_err.mean().item()
    print(f"  {'ALL':>8} {tf_all:10.6f} {fr_all:10.6f} {fr_all / max(tf_all, 1e-12):7.1f}x")

    # 找每条 episode 里 free-run 误差首次离开教师强制误差带的 tick：
    # 阈值取"教师强制误差中位数的 10 倍"（1e-9 兜底防止除零式的退化）
    thresh = 10.0 * max(tf_err.median().item(), 1e-9)
    first_div = []
    for e in range(n_ep):
        idx = (fr_err[e] > thresh).nonzero()
        # 行号 * 2 = tick；整条都没超阈值就记 ROWS*2（=830，表示从未发散）
        first_div.append(int(idx[0]) * 2 if len(idx) else ROWS * 2)
    first_div = np.array(first_div)
    print(f"[diag] free-run error first exceeds 10x teacher median at tick "
          f"median={int(np.median(first_div))} (p25={int(np.percentile(first_div, 25))}, "
          f"p75={int(np.percentile(first_div, 75))}; {int((first_div >= ROWS * 2).sum())}/{n_ep} "
          f"episodes never diverge)")
    print("[diag] verdict guide: free-run/teacher ratio >~5x in DESCEND..CLOSE = the net "
          "depends on last_action (shortcut confirmed); ~1x throughout = shortcut absent, "
          "look elsewhere (state coverage / multi-modality).")

    # ---- 测试 3：置换重要性（在随机采样的行上做） ----
    # 原理：把某个观测块在行与行之间随机打乱（该块与动作的对应关系被破坏，但边缘分布不变），MSE 涨得越多，说明网络越依赖这个块。
    n_rows = obs.shape[0]
    sample = torch.randperm(n_rows)[: min(args.perm_rows, n_rows)]
    o_s = torch.as_tensor(obs, dtype=torch.float32)[sample].to(args.device)
    a_s = torch.as_tensor(act, dtype=torch.float32)[sample].to(args.device)
    with torch.no_grad():
        base = ((model(o_s) - a_s) ** 2).mean().item()   # 不打乱时的基线 MSE
        print(f"\n[diag] permutation importance (sample {len(sample)} rows, baseline MSE {base:.6f})")
        results = []
        for name, blk in BLOCKS.items():
            o_p = o_s.clone()
            # 只打乱这一个块：用随机行序重排后的同一块数据覆盖
            o_p[:, blk] = o_s[torch.randperm(len(sample), device=args.device)][:, blk]
            mse = ((model(o_p) - a_s) ** 2).mean().item()
            results.append((mse - base, name))
        # 按 MSE 增量从大到小排序输出（增量越大 = 网络越依赖该块）
        for delta, name in sorted(results, reverse=True):
            print(f"  {name:>12}: +{delta:.6f}  ({delta / max(base, 1e-12):6.1f}x baseline)")
    print("[diag] verdict guide: last_action towering over obj_rel_R/place_rel = shortcut; "
          "obj_rel_R/place_rel meaningfully high = the net does read the piece state.")


if __name__ == "__main__":
    main()
