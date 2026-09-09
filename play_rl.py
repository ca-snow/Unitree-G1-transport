# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
在抓放环境中回放"经 PPO 微调"的 skrl checkpoint，并统计成功率。
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description="Evaluate a fine-tuned skrl PPO policy on the pickplace task.")
# --task：任务名，默认本项目的抓放任务
parser.add_argument("--task", type=str, default="Isaac-PickPlace-RL-G129-Dex3-v0")
# --checkpoint：skrl 智能体 checkpoint 路径（必填），如 best_agent.pt
parser.add_argument("--checkpoint", type=str, required=True, help="skrl agent checkpoint (e.g. best_agent.pt)")
# --num_envs：并行环境数。默认 32，统计成功率时样本量足够
parser.add_argument("--num_envs", type=int, default=32)
# --rollouts：重复评估轮数。总样本数 = num_envs × rollouts（默认 32×4=128）
parser.add_argument("--rollouts", type=int, default=4)
# --horizon：每轮走多少个控制步。725 是刻意取的：要小于环境自身的
# 800 步超时限制，否则环境会中途自动 reset，打乱评估
parser.add_argument("--horizon", type=int, default=725, help="Steps per rollout (< 800-step env timeout).")
# --record_video：加上此开关则录制 mp4
parser.add_argument("--record_video", action="store_true")
# --seed：随机种子；None 表示用配置默认
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录视频要开渲染相机
if args_cli.record_video:
    args_cli.enable_cameras = True

# 剩余参数交还 hydra 解析
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Isaac Sim（必须先于其他 isaaclab import）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

# 云端项目根目录
PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import gymnasium as gym

import tasks.rl.g1_pickplace
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config

# 复用 BC 训练脚本里定义的网络结构（策略+价值共享主干），保证结构一致才能加载权重
from bc_train import SharedPolicy

# 成功判据：物体最终高度比初始静置高度高出 0.05 米（5 厘米）就算"抓起成功"。
# 这是早期"只评抓取"的旧判据，为了兼容旧 checkpoint 的评估结果而保留。
SUCCESS_LIFT = 0.05

# 脚本专家演示的各阶段结束时刻（控制步序号 -> 阶段名），用于在对应时刻打印遥测：
# HOLD 保持 / UP 抬臂 / TRAV 平移 / DESCEND 下降 / INSERT 手指探入 /
# CLOSE 合拢 / LIFT 抬起 / CARRY 携带 / LOWER 放低 / RELEASE 松开 / RETREAT 撤回
PHASE_CHECKPOINTS = {
    19: "HOLD", 64: "UP", 104: "TRAV", 184: "DESCEND", 244: "INSERT",
    324: "CLOSE", 404: "LIFT", 504: "CARRY", 584: "LOWER",
    634: "RELEASE", 724: "RETREAT",
}
# 观测向量（共 110 维）布局：
# 关节位置 0:29 | 关节速度 29:58 | 手部关节 58:72 |
# 物体相对左腕 72:75 | 物体相对右腕 75:78 | 上一步动作 78:106 |
# 阶段标量 106 | 放置目标相对位置 107:110
# 下面这个切片取"物体相对右腕的三维位移"，用来打印手到物体的距离
RIGHT_OBJ_REL_SLICE = slice(75, 78)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    #创建环境、加载微调后的策略权重、跑若干轮并统计抓起成功率。

    #env_cfg / agent_cfg 由 hydra 装饰器自动加载注入（agent_cfg 这里其实
    #用不上，这里绕过skrl Runner直接用SharedPolicy加载权重）。

    device = args_cli.device or env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        torch.manual_seed(args_cli.seed)  # 环境和 torch 全局种子都固定，保证可复现

    # 相机视角：固定在世界坐标系里、正对操作台的机位（这几组数是调好的观察位）
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (-3.55, -3.55, 1.25)     # 相机位置（米）
    env_cfg.viewer.lookat = (-4.24, -4.02, 0.85)  # 相机注视点（米）

    # 只有录像时才需要 rgb_array 渲染，否则不渲染节省资源
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.record_video else None)
    if args_cli.record_video:
        video_dir = os.path.join(os.getcwd(), "rl_videos")
        # 一条视频录完全部轮次：总帧数 = 每轮步数 × 轮数
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.horizon * args_cli.rollouts,
            disable_logger=True,
        )
        print(f"[rl-play] recording video to: {video_dir}")

    base_env = env.unwrapped
    obj = base_env.scene["object"]  # 待抓取的工件（用它的高度判成功）

    obs_dict, _ = env.reset()
    obs_dim = obs_dict["policy"].shape[-1]                  # 观测维度（应为 110）
    act_dim = base_env.action_manager.total_action_dim      # 动作维度（应为 28）

    # skrl 的 checkpoint 按模块分别保存 state_dict；由于策略和价值网络共享
    # 主干（shared model），"policy" 这一项就包含了整张网络（含价值头），
    # 恰好与 bc_train.py 里 SharedPolicy 的参数布局一致，可以直接加载。
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    state = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt["model"]
    policy = SharedPolicy(obs_dim, act_dim).to(device)
    # strict=False：允许缺少价值头（value_layer）参数——评估只用策略头，
    # 但除价值头外若还缺别的参数、或多出参数，就是结构对不上，必须报错
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if [k for k in missing if not k.startswith("value_layer")] or unexpected:
        raise RuntimeError(f"checkpoint/model mismatch: missing={missing} unexpected={unexpected}")
    policy.eval()  # 评估模式（关闭 dropout 等训练期行为）
    print(f"[rl-play] loaded {args_cli.checkpoint}")

    # 记录物体初始静置高度：成功 = 最终高度比它高出 SUCCESS_LIFT（5 厘米）
    rest_z = obj.data.root_pos_w[:, 2].mean().item()
    total_success, total_envs = 0, 0

    # ---------- 评估主循环：rollouts 轮 × horizon 步 ----------
    for r in range(args_cli.rollouts):
        if r > 0:
            obs_dict, _ = env.reset()  # 第 2 轮起每轮先重置环境
        for t in range(args_cli.horizon):
            with torch.no_grad():  # 纯推理，不需要梯度
                action = policy(obs_dict["policy"].to(device))
            obs_dict, _, _, _, _ = env.step(action)
            # 在每个阶段结束的时刻打印遥测：手到物体的平均距离 + 物体平均高度，
            # 出问题时能一眼看出策略是在哪个阶段跑偏的
            if t in PHASE_CHECKPOINTS:
                rel = obs_dict["policy"][:, RIGHT_OBJ_REL_SLICE]
                print(f"[rl-play]   t={t:3d} after {PHASE_CHECKPOINTS[t]:7s} "
                      f"hand->obj={rel.norm(dim=-1).mean().item():.3f} m | "
                      f"obj_z={obj.data.root_pos_w[:, 2].mean().item():.3f} m")
        # 本轮结束：逐环境判定"物体是否被抬高超过阈值"
        final_z = obj.data.root_pos_w[:, 2]
        success = (final_z - rest_z) > SUCCESS_LIFT
        total_success += success.sum().item()
        total_envs += base_env.num_envs
        print(f"[rl-play] rollout {r + 1}/{args_cli.rollouts}: "
              f"success {success.sum().item()}/{base_env.num_envs} "
              f"| obj_z(mean)={final_z.mean().item():.3f} m")

    # 汇总所有轮次的总成功率
    print(f"[rl-play] TOTAL: {total_success}/{total_envs} = {total_success / total_envs:.2%}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
