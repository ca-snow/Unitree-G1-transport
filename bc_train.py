# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
行为克隆（BC, Behaviour Cloning）训练器 —— 抓放技能管线的第 2 步（共 4 步）。
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn


LAST_ACTION_OBS = slice(78, 106)

class SharedPolicy(nn.Module):
    #skrl 共享"高斯策略 + 价值"模型的镜像实现。

    #输入：obs 张量，形状 [batch, obs_dim]（本项目 obs_dim=110）。
    #输出：确定性动作（即高斯分布的均值），形状 [batch, act_dim]（act_dim=28）。


    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # 共享主干：110 -> 256 -> 128 -> 128，激活函数 ELU
        # （与 skrl_ppo_cfg.yaml 里的 net_container 完全一致）
        self.net_container = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
        )
        # 策略头：128 -> 28，输出高斯分布的均值
        self.policy_layer = nn.Linear(128, act_dim)
    
        self.value_layer = nn.Linear(128, 1)
        # 高斯策略的对数标准差参数（PPO 探索用），BC 阶段保持初始 0 不动
        self.log_std_parameter = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        #前向：obs [batch, 110] -> 确定性动作 [batch, 28]（只走策略头）。
        return self.policy_layer(self.net_container(obs))


def main():
    #训练入口：加载 .npz 数据集 -> episode 级切分 -> MSE 回归训练 -> 保存最优检查点。
    parser = argparse.ArgumentParser(description="Behaviour cloning on the grasp-expert dataset.")
   
    parser.add_argument(
        "--dataset", type=str, required=True)
    # --out：输出检查点（.pt）路径。
    parser.add_argument("--out", type=str, required=True)
    # --epochs：完整遍历训练集的轮数；默认 60，与下面余弦退火调度的
    # T_max=epochs 配套，训练结束时学习率恰好退到最低。
    parser.add_argument("--epochs", type=int, default=60)
    # --batch_size：每个梯度步的样本行数。网络很小（[256,128,128]），
    # 4096 的大批量在 GPU 上依然很快，且梯度估计更平稳。
    parser.add_argument("--batch_size", type=int, default=4096)
    # --lr：Adam 初始学习率 1e-3，随余弦退火逐渐降低。
    parser.add_argument("--lr", type=float, default=1e-3)
    # --val_frac：留出 5% 数据做验证（按默认的 episode 级切分，是 5% 的"整条轨迹"）。
    parser.add_argument("--val_frac", type=float, default=0.05)
    # --episode_rows：episode 级 train/val 切分时，每条 episode 占多少行。
    # transport 采集节奏：一个完整周期 830 个 BC tick，每 2 个 tick 存 1 行
    # => 830 / 2 = 415 行/episode，故默认 415。
    parser.add_argument("--episode_rows", type=int, default=415)
    # --lastact_dropout：每个【训练】样本以该概率把 last_action 观测块
    # （第 78:106 维）整块置零。
    parser.add_argument("--lastact_dropout", type=float, default=0.0)
    # --seed：随机种子（切分、打乱、权重初始化全部可复现）。
    parser.add_argument("--seed", type=int, default=42)
    # --device：有 CUDA 就用 GPU，否则退回 CPU。
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- 加载数据集：多个 .npz 按行拼接成一个大数组 ----
    obs_list, act_list = [], []
    for path in args.dataset.split(","):
        data = np.load(path.strip())
        obs_list.append(data["obs"])
        act_list.append(data["actions"])
        print(f"[bc]   loaded {data['obs'].shape[0]} pairs from {path.strip()}")
    obs = torch.as_tensor(np.concatenate(obs_list, axis=0), dtype=torch.float32)
    act = torch.as_tensor(np.concatenate(act_list, axis=0), dtype=torch.float32)
    assert obs.shape[0] == act.shape[0], "obs/actions row count mismatch"
    n, obs_dim = obs.shape
    act_dim = act.shape[1]
    print(f"[bc] dataset: {n} pairs | obs_dim={obs_dim} | act_dim={act_dim}")

    # ---- train/val 切分：默认按 EPISODE（整条轨迹）级切分 ----
    if args.episode_rows > 0 and n % args.episode_rows == 0:
        n_ep = n // args.episode_rows                       # episode 总数
        ep_perm = torch.randperm(n_ep)                      # 随机打乱 episode 顺序
        n_val_ep = max(1, int(n_ep * args.val_frac))        # 验证 episode 数（至少 1 条）
        rows = torch.arange(n).reshape(n_ep, args.episode_rows)  # [n_ep, 415] 每行是一条 episode 的行号
        val_idx = rows[ep_perm[:n_val_ep]].reshape(-1)      # 前 n_val_ep 条 episode 的全部行 -> 验证集
        train_idx = rows[ep_perm[n_val_ep:]].reshape(-1)    # 其余 episode 的全部行 -> 训练集
        print(f"[bc] episode-level split: {n_ep} episodes -> "
              f"train {n_ep - n_val_ep} ep / val {n_val_ep} ep")
    else:
        # 行数不是 episode_rows 的整数倍（或用户显式传 0）：退回行级切分。
        # 注意此时 val MSE 会因数据泄漏而"虚低"，只能参考不能当真。
        if args.episode_rows > 0:
            print(f"[bc] WARNING: {n} rows is not a multiple of --episode_rows "
                  f"{args.episode_rows} - falling back to the ROW-level split "
                  f"(val MSE will read optimistically)")
        perm = torch.randperm(n)
        n_val = max(1, int(n * args.val_frac))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
    obs_train, act_train = obs[train_idx], act[train_idx]
    obs_val, act_val = obs[val_idx].to(args.device), act[val_idx].to(args.device)
    print(f"[bc] train={len(train_idx)} val={len(val_idx)}")

    # last_action dropout 只对 110 维观测布局有意义
    # 维度不对说明数据集来自别的布局，直接禁用以免置零到错误的字段。
    if args.lastact_dropout > 0.0 and obs_dim != 110:
        print(f"[bc] WARNING: --lastact_dropout assumes the 110-dim obs layout, "
              f"got obs_dim={obs_dim} - disabled")
        args.lastact_dropout = 0.0
    if args.lastact_dropout > 0.0:
        print(f"[bc] last-action obs dropout active: p={args.lastact_dropout} "
              f"(dims {LAST_ACTION_OBS.start}:{LAST_ACTION_OBS.stop} zeroed per training sample)")

    model = SharedPolicy(obs_dim, act_dim).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 余弦退火：学习率从 args.lr 平滑降到接近 0，T_max=epochs 即一个完整周期
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    loss_fn = nn.MSELoss()   # 行为克隆核心：动作的均方误差回归

    best_val = float("inf")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm_e = torch.randperm(len(train_idx))  # 每轮重新打乱训练行的顺序
        total, count = 0.0, 0
        for i in range(0, len(perm_e), args.batch_size):
            idx = perm_e[i:i + args.batch_size]
            xb = obs_train[idx].to(args.device, non_blocking=True)  # 本批观测 [B,110]
            yb = act_train[idx].to(args.device, non_blocking=True)  # 本批目标动作 [B,28]
            if args.lastact_dropout > 0.0:
                # 按样本抽签：drop 为 [B] 的布尔掩码，True 的样本把last_action 观测块（78:106）整块置零

                drop = torch.rand(xb.shape[0], device=xb.device) < args.lastact_dropout
                xb[drop, LAST_ACTION_OBS] = 0.0
            pred = model(xb)             # 前向：预测动作 [B,28]
            loss = loss_fn(pred, yb)     # 与专家动作的 MSE
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += loss.item() * len(idx)   # 累计加权损失（按样本数加权）
            count += len(idx)
        sched.step()   # 每个 epoch 结束后走一步余弦退火

        # ---- 每轮结束在验证集上评估，只保存 val MSE 最优的检查点 ----
        model.eval()
        with torch.no_grad():
            val_mse = loss_fn(model(obs_val), act_val).item()
        marker = ""
        if val_mse < best_val:
            best_val = val_mse
            torch.save(
                {
                    "model": model.state_dict(),          # 全部权重（含价值头和 log_std，供 PPO 热启动）
                    "obs_dim": obs_dim,                   # 观测维度（加载方用它重建网络，=110）
                    "act_dim": act_dim,                   # 动作维度（=28）
                    "val_mse": val_mse,                   # 保存时的验证 MSE（供下游打印/追溯）
                    "dataset": os.path.abspath(args.dataset),        # 训练数据来源（溯源用）
                    "lastact_dropout": args.lastact_dropout,         # 训练时用的 dropout 概率（溯源用）
                },
                args.out,
            )
            marker = "  <- saved"
        print(f"[bc] epoch {epoch:3d}/{args.epochs} | train MSE {total / count:.6f} | "
              f"val MSE {val_mse:.6f}{marker}")

    print(f"[bc] done. best val MSE = {best_val:.6f} -> {args.out}")


if __name__ == "__main__":
    main()
