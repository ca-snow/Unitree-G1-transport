# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
用 skrl 训练强化学习智能体的入口脚本（G1 抓放任务专用版）。
"""

import argparse, sys

from isaaclab.app import AppLauncher

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description="Train RL agent with skrl.")
# --video：训练期间是否周期性录制视频（用来远程观察训练进展）
parser.add_argument("--video", action="store_true", default=False)
# video_length：每段视频录多少帧。
# video_interval：每隔 N 个"环境步"（env step）触发一次录制——这是env.step 的计数
#一次 PPO 迭代 = `rollouts` 个环境步（rollouts 在 agent 的 yaml 配置里），如果按迭代数思考，要自己换算。
# 录像用的是仿真视口（viewport）渲染画面（RecordVideo 包装器），与机器人身上的机载相机无关、互不影响。
parser.add_argument("--video_length", type=int, default=300)
parser.add_argument("--video_interval", type=int, default=5000)
# --num_envs：并行环境数；None 表示用环境配置文件里的默认值
parser.add_argument("--num_envs", type=int, default=None)
# --task：要训练的任务名（如 Isaac-PickPlace-RL-G129-Dex3-v0）
parser.add_argument("--task", type=str, default=None)
# --agent：手动指定 skrl 配置入口名；None 表示按 --algorithm 自动推断
parser.add_argument("--agent", type=str, default=None)
# --seed：随机种子；None 用配置默认，-1 表示随机抽一个
parser.add_argument("--seed", type=int, default=None)
# --distributed：多 GPU 分布式训练开关
parser.add_argument("--distributed", action="store_true", default=False)
# --checkpoint：从已有 checkpoint 继续训练（断点续训 / 微调）
parser.add_argument("--checkpoint", type=str, default=None)
# --max_iterations：最大 PPO 迭代数；None 用 yaml 里配置的 timesteps
parser.add_argument("--max_iterations", type=int, default=None)
# --export_io_descriptors：导出观测/动作的结构描述文件（部署对接用）
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
# --ml_framework：skrl 后端框架，本项目一直用 torch
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch","jax","jax-numpy"])
# --algorithm：训练算法，本项目用 PPO
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP","PPO","IPPO","MAPPO"])
# --ray-proc-id：Ray 集群批量调参时的进程编号（单机训练用不到）
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# 要录视频必须开渲染相机
if args_cli.video:
    args_cli.enable_cameras = True

# argparse 认不出的参数交还给 hydra 解析（可用来覆盖 env/agent 配置项）
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Isaac Sim（必须先于所有 isaaclab 子模块的 import）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 通过包导入注册自定义 G1 抓放任务 ----
# 任务在 tasks/rl/g1_pickplace/__init__.py 里自己调用 gym.register 完成注册，
# 所以只要 import 这个包，任务名就能被 gym.make 找到。
import os
import sys

# 云端服务器上 unitree 仿真项目的根目录
PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"

# 原版仿真是通过 sim_main.py 启动的，它会执行os.environ["PROJECT_ROOT"] = <项目目录>
# 而 common_scene / 机器人配置模块是在"被 import 的那一刻"读取os.environ.get("PROJECT_ROOT") 来拼 USD 资产路径的
#我们是经 isaaclab.sh 启动、不走 sim_main.py，所以必须在 import 任务包之前在这里手动设置好，
# 否则资产路径会变成 "None/assets/..." 而加载失败。
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

# 把项目根目录加进模块搜索路径，让 `tasks` 包可以被 import
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import tasks.rl.g1_pickplace
import tasks.rl.g1_locomotion

#以下是官方训练脚本的原有流程。

import logging, os, random, time
from datetime import datetime
import gymnasium as gym2
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

from isaaclab.envs import (
    DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg,
    ManagerBasedRLEnvCfg, multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

# 确定 skrl 智能体配置的入口点名字：
#   PPO 用通用的 "skrl_cfg_entry_point"，其他算法用 "skrl_<算法名>_cfg_entry_point"；
#   若用户用 --agent 显式给了入口名，则反过来从入口名解析出算法名
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg, agent_cfg):
    #训练主流程：整理配置 -> 建日志目录 -> 创建环境 -> skrl Runner 跑训练。

    #env_cfg / agent_cfg 由 hydra 装饰器根据任务注册信息自动加载注入，
    #命令行参数在函数体内逐项覆盖它们。
    # 命令行参数优先，没给就用配置文件里的默认值
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # 分布式训练只支持 GPU；每个进程用自己的本地 GPU 编号
    if args_cli.distributed and args_cli.device and "cpu" in args_cli.device:
        raise ValueError("Distributed training not supported on CPU.")
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # skrl 的 trainer 以"环境步（timesteps）"计数，而用户给的是"迭代数"：
    # 一次 PPO 迭代 = rollouts 个环境步，这里做换算
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    # 环境的关闭由本脚本自己负责（最后的 env.close()），不让 trainer 代劳
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # seed=-1表示随机抽一个种子（0~10000）
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed or agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]  # 环境和智能体种子保持一致

    # ---------- 日志目录：logs/skrl/<实验目录>/<时间戳>_<算法>_<框架>[_<实验名>] ----------
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # 把最终目录写回配置，skrl 内部（tensorboard、checkpoint 保存）也会用它
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # 把本次训练用的 env/agent 配置快照存盘，事后可以精确复现
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 断点续训：把 --checkpoint 解析成本地文件路径（支持远程 URL 自动下载）
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env_cfg.log_dir = log_dir
    # 只有录视频时才需要 rgb_array 渲染
    env = gym2.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # 多智能体环境 + PPO（单智能体算法）时先转成单智能体接口（本任务用不到，保险起见保留）
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # 训练期周期性录像：每 video_interval 个环境步录一段 video_length 帧的视频
    if args_cli.video:
        env = gym2.wrappers.RecordVideo(env, **{
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        })

    start_time = time.time()
    # 包成 skrl 的向量化环境接口，交给 Runner（它按 agent_cfg 建网络/优化器/训练器）
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    runner = Runner(env, agent_cfg)

    # 断点续训：先把旧权重灌进去再开始训练
    if resume_path:
        runner.agent.load(resume_path)

    runner.run()  # 正式开始训练（内部循环直到 timesteps 用完）
    print(f"Training time: {round(time.time() - start_time, 2)} s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
