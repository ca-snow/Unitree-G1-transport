# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
回放 / 评估 G1 抓放任务上训练好的 skrl 智能体（train-catch.py 的配套回放脚本）。
"""

import argparse, sys

from isaaclab.app import AppLauncher

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description="Play a trained skrl agent (G1 pickplace).")
# --checkpoint：要回放的 .pt 权重文件路径（必填），一般选 checkpoints/best_agent.pt
parser.add_argument("--checkpoint", type=str, required=True)
# --num_envs：可视化用的并行环境数。默认 16——够看出统计趋势又不至于画面太挤/太占显存
parser.add_argument("--num_envs", type=int, default=16)
# --video_length：录制帧数。默认 500 帧 = 5 秒（抓放任务控制频率 100Hz，1 帧对应 1 个控制步）
parser.add_argument("--video_length", type=int, default=500)
# --task：任务名，默认本项目的抓放任务
parser.add_argument("--task", type=str, default="Isaac-PickPlace-RL-G129-Dex3-v0")
# --agent：手动指定 skrl 配置入口名；默认 None 表示按 --algorithm 自动推断
parser.add_argument("--agent", type=str, default=None)
# --seed：随机种子；None 表示沿用配置文件里的种子
parser.add_argument("--seed", type=int, default=None)
# --ml_framework：skrl 的后端框架，本项目一直用 torch
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
# --algorithm：算法名，本项目训练用的是 PPO
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 无显示器服务器上要录视频就必须开启渲染（相机），这里强制打开
args_cli.enable_cameras = True

# 把 argparse 不认识的参数还给 hydra 解析
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Isaac Sim（必须先于所有 isaaclab 子模块的 import）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 通过包导入注册自定义 G1 抓放任务（与训练脚本 train-catch.py 完全一致）----
import os
import sys

# 云端项目根目录
PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tasks.rl.g1_pickplace
import tasks.rl.g1_locomotion

import os
import gymnasium as gym2
import torch
import skrl
from packaging import version

# skrl 1.4.3 之前的版本 Runner 接口不兼容，直接拒绝运行
if version.parse(skrl.__version__) < version.parse("1.4.3"):
    skrl.logger.error("skrl version >= 1.4.3 required")
    exit()

# 按后端框架选择对应的 Runner（本项目只用 torch 分支）
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

# 确定 skrl 智能体配置的入口点名字：
#   PPO 用通用的 "skrl_cfg_entry_point"，其他算法用 "skrl_<算法名>_cfg_entry_point"；
#   若用户用 --agent 显式给了入口名，则反过来从入口名里解析出算法名
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):
    #创建环境、加载 checkpoint、以确定性策略跑一段并录制视频。

    #env_cfg / agent_cfg 由 hydra 装饰器根据任务注册信息自动加载注入。
    
    # 命令行参数优先，没给再用配置文件里的默认值
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # 种子同步：环境和智能体必须用同一个种子，结果才可复现
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # 视频输出目录放在 checkpoint 旁边（同一次训练 run 的目录下），方便对应查找：
    # .../<run>/checkpoints/x.pt -> 向上两级得到 <run>，视频存 <run>/videos/play
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    video_dir = os.path.join(run_dir, "videos", "play")

    # render_mode="rgb_array"：让环境输出像素帧，供录像包装器抓取
    env = gym2.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # 多智能体环境 + PPO（单智能体算法）时需要先转成单智能体接口（本任务用不到，保险起见保留）
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # 录像包装器：step==0时触发一次录制，共录 video_length 帧
    env = gym2.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    print(f"[play] recording video to: {video_dir}")

    # 包成 skrl 认识的向量化环境接口
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # 用 skrl Runner 按配置重建网络结构，再把 checkpoint 权重灌进去
    runner = Runner(env, agent_cfg)
    print(f"[play] loading checkpoint: {ckpt_path}")
    runner.agent.load(ckpt_path)
    runner.agent.set_running_mode("eval")  # 切到评估模式（关闭探索/训练行为）

    # ---------- 回放主循环 ----------
    obs, _ = env.reset()
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():  # 纯推理，不记录梯度，省显存提速
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # 优先取动作分布的均值（确定性动作）；没有 mean_actions 就退回采样值
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
        timestep += 1
        if timestep >= args_cli.video_length:
            break  # 录满预定帧数即结束

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
