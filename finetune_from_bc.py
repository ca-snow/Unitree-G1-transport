# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
用行为克隆权重作为初始值的 PPO 微调 —— 管线第 4 步（共 4 步）。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO fine-tuning from a BC checkpoint.")
# --task：训练任务 id（G1 29 关节 + Dex3 手的抓放任务）。
parser.add_argument("--task", type=str, default="Isaac-PickPlace-RL-G129-Dex3-v0")
# --bc_checkpoint：bc_train.py 产出的 bc_policy.pt（必填，热启动来源）。
parser.add_argument("--bc_checkpoint", type=str, required=True)
# --num_envs：并行环境数（PPO 需要大批量采样，默认 2048）。
parser.add_argument("--num_envs", type=int, default=2048)
# --timesteps：覆盖 yaml 里的 trainer 总步数（None = 用 yaml 默认值）。
parser.add_argument("--timesteps", type=int, default=None)
# --learning_rate：初始学习率。从零训练的配置用 1e-3；这里降到 1e-4，
# 更低的学习率能保护克隆来的行为不被早期更新冲垮。
parser.add_argument("--learning_rate", type=float, default=1.0e-4)
# --initial_log_std：策略初始对数标准差。-2.5 ≈ std 0.08 动作单位
# ≈ 每关节每步 0.04 rad；旧值 -1.0（≈0.18 rad）的探索噪声直接毁掉抓取。
parser.add_argument("--initial_log_std", type=float, default=-2.5)
# --critic_warmup：冻结"主干+策略头+log_std"、只训练价值头的环境步数。
# 4800 = 在 rollouts=24 的设置下正好 200 次 PPO 更新；传 0 关闭预热。
parser.add_argument("--critic_warmup", type=int, default=4800)
# --seed：随机种子（None = 不固定）。
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 剩余参数交给 hydra（Isaac Lab 配置系统）解析
sys.argv = [sys.argv[0]] + hydra_args

# 必须先启动 Isaac Sim 应用，之后才能 import isaaclab 的其余模块
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

# 项目根目录：加入 sys.path 才能 import 本项目的 tasks 包（任务注册代码）
PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import gymnasium as gym

from skrl.utils.runner.torch import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import tasks.rl.g1_pickplace
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    #微调主函数：改写 PPO 配置 -> 建环境 -> 移植 BC 权重 -> critic 预热 -> 训练。

    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg["seed"] = args_cli.seed

    # --- 在 skrl_ppo_cfg.yaml 基础上叠加微调专用覆盖项 ---
    agent_cfg["agent"]["learning_rate"] = args_cli.learning_rate
    agent_cfg["models"]["policy"]["initial_log_std"] = args_cli.initial_log_std
    agent_cfg["agent"]["entropy_loss_scale"] = 0.0
    # 给 KL 自适应学习率设上限
    sched_kwargs = agent_cfg["agent"].setdefault("learning_rate_scheduler_kwargs", {})
    sched_kwargs["min_lr"] = 1.0e-6   # 学习率下限
    sched_kwargs["max_lr"] = 3.0e-4   # 学习率上限（远低于 skrl 默认的 1e-2）
    if args_cli.timesteps is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.timesteps

    from datetime import datetime

    log_root = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"] or "g1_pickplace")
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_finetune_from_bc"
    agent_cfg["agent"]["experiment"]["directory"] = os.path.abspath(log_root)
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    print(f"[finetune] logging to {os.path.join(os.path.abspath(log_root), run_name)}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")   # 包装成 skrl 认识的向量化环境

    runner = Runner(env, agent_cfg)   # skrl Runner：按配置组装 PPO 智能体 + 训练器

    # --- 把 BC 权重移植进共享的策略/价值模型 ---
    ckpt = torch.load(args_cli.bc_checkpoint, map_location=env.device)
    policy_model = runner.agent.models["policy"]
    missing, unexpected = policy_model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        # 检查点里有 skrl 模型不认识的键：两边结构漂移了，立刻报错
        raise RuntimeError(f"BC checkpoint has unexpected keys: {unexpected}")
    # missing 也必须为空（bc_train.SharedPolicy 与 skrl 实例化的模型是精确镜像）；
    if missing:
        raise RuntimeError(f"skrl model keys not covered by the BC checkpoint: {missing}")
    assert runner.agent.models["value"] is policy_model, "expected a shared policy/value model"
    print(f"[finetune] loaded BC weights from {args_cli.bc_checkpoint} "
          f"(val MSE {ckpt.get('val_mse', float('nan')):.6f})")
    print(f"[finetune] lr={args_cli.learning_rate} initial_log_std={args_cli.initial_log_std} "
          f"critic_warmup={args_cli.critic_warmup} entropy=0")

    # --- critic 预热：冻结价值头以外的全部参数，让随机初始化的 critic
    # 无法把梯度推过共享主干、摧毁克隆好的策略。预热步数走完后解冻。
    if args_cli.critic_warmup > 0:
        # 名字不以 value_layer 开头的参数全部冻结（主干+策略头+log_std）
        frozen = [p for n, p in policy_model.named_parameters() if not n.startswith("value_layer")]
        for p in frozen:
            p.requires_grad_(False)
        print(f"[finetune] critic warm-up: {len(frozen)} policy tensors frozen "
              f"for the first {args_cli.critic_warmup} env-steps")

        # 每个环境步之后检查是否到达预热步数，到了就解冻并恢复正常训练。
        original_post = runner.agent.post_interaction
        state = {"unfrozen": False}   # 闭包里的解冻标志（只解冻一次）

        def post_interaction(timestep, timesteps):
            if not state["unfrozen"] and timestep >= args_cli.critic_warmup:
                for p in frozen:
                    p.requires_grad_(True)   # 解冻：策略从此开始接受 PPO 梯度
                state["unfrozen"] = True
                print(f"[finetune] critic warm-up done at step {timestep}: policy unfrozen")
            return original_post(timestep, timesteps)

        runner.agent.post_interaction = post_interaction

    runner.run("train")   # 进入标准 skrl PPO 训练循环
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
