# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
一次性检查 G1 抓放（pickplace）环境的运动学信息（只打印、不训练）。
"""

import argparse, sys

from isaaclab.app import AppLauncher

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description="Inspect G1 pickplace env kinematics.")
# --task：要检查的 gym 任务名。默认就是本项目的抓放任务（G1 二十九关节 + Dex3 手）。
parser.add_argument("--task", type=str, default="Isaac-PickPlace-RL-G129-Dex3-v0")
# --num_envs：并行环境数量。检查关节信息只需要 1 个环境，多了纯属浪费显存。
parser.add_argument("--num_envs", type=int, default=1)
# 把 Isaac Lab 启动器自带的参数（--headless、--device 等）也挂到解析器上。
AppLauncher.add_app_launcher_args(parser)
# parse_known_args：识别不了的参数留给 hydra（Isaac Lab 的配置系统）去处理。
args_cli, hydra_args = parser.parse_known_args()
# 把剩余参数重新塞回 sys.argv，供后面 hydra 解析；否则 hydra 会报"未知参数"。
sys.argv = [sys.argv[0]] + hydra_args


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import sys


PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
# 任务的场景配置在 import 时会读取环境变量 PROJECT_ROOT 来拼 USD 资产路径，
# 所以必须在 import 任务包之前把它设置好。
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)  # 让 `tasks` 包可以被 import

import torch
import gymnasium as gym2
import tasks.rl.g1_pickplace
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    #创建1个抓放环境并打印全部运动学信息（关节限位、连杆名、物体到手腕距离）。

    #参数由 hydra 装饰器自动注入：
    #  env_cfg   —— 任务的环境配置对象（场景、机器人、物体等）；
    #  agent_cfg —— skrl 智能体配置（本脚本用不到，但装饰器要求签名里有）。
    
    env_cfg.scene.num_envs = args_cli.num_envs  # 覆盖为命令行指定的环境数（默认 1）
    # render_mode=None：不渲染画面，纯物理仿真即可（只需要数据）
    env = gym2.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env.reset()  # 必须 reset 一次，机器人/物体数据缓冲区才会被填充

    # env.unwrapped 拿到最里层的 Isaac Lab 环境，从场景里取出机器人和待抓物体
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]

    # --- 第一部分：打印所有关节（名字 | 下限 | 上限 | 默认值） ---
    names = robot.data.joint_names
    limits = None
    # 不同 Isaac Lab 版本里关节限位属性的名字不一样，按新到旧依次尝试三个候选名
    for attr in ("joint_pos_limits", "soft_joint_pos_limits", "joint_limits"):
        if hasattr(robot.data, attr):
            limits = getattr(robot.data, attr)[0]  # 取第 0 个环境，形状 [关节数, 2]（下限/上限）
            print(f"(joint limits read from robot.data.{attr})")
            break
    if limits is None:
        raise RuntimeError("Could not find joint limits attribute on robot.data")
    default = robot.data.default_joint_pos[0]     # 默认关节角，形状 [关节数]
    print("\n================ JOINTS (name | lower | upper | default) ================")
    for i, n in enumerate(names):
        lo = float(limits[i, 0]); hi = float(limits[i, 1]); d = float(default[i])
        tag = ""
        if "hand" in n:
            # 手指"合拢方向"提示：默认姿态是张开的，离默认值更远的那一端限位
            # 就是合拢方向（UPPER/+ 表示往上限即正方向转是抓紧，LOWER/- 反之）
            tag = "   <-- FINGER (close toward %s)" % ("UPPER/+" if abs(hi - d) > abs(d - lo) else "LOWER/-")
        print(f"[{i:2d}] {n:32s} lo={lo:+.3f} hi={hi:+.3f} def={d:+.3f}{tag}")

    # --- 第二部分：Dex3 手指汇总 ---
    # 关节顺序已对照 unitree 官方 xr_teleoperate 仓库确认：
    #   teleop/robot_control/robot_hand_unitree.py + hand_retargeting.py。
    # 每只手 7 个自由度：拇指 thumb(0,1,2) + 中指 middle(0,1) + 食指 index(0,1)。
    # 张开姿态 = 全 0，因此"合拢"就是把每个关节推向离 0 更远的那个限位端。
    dex3_left = [
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
    ]
    dex3_right = [
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
    ]
    # 建立"关节名 -> 索引"的映射，方便按名字查表
    name_to_idx = {n: i for i, n in enumerate(names)}
    for side, jn in [("LEFT", dex3_left), ("RIGHT", dex3_right)]:
        print(f"\n========= DEX3 {side} FINGERS (grasp target = 'close' col) =========")
        closed = []  # 收集这只手 7 个关节的"合拢目标角"，直接可用于抓取脚本
        for n in jn:
            i = name_to_idx.get(n)
            if i is None:
                # 该 USD 模型里没有这个关节（比如换了手的版本），提示并占位 0
                print(f"  {n:28s} <-- NOT FOUND in this USD")
                closed.append(0.0)
                continue
            lo = float(limits[i, 0]); hi = float(limits[i, 1]); d = float(default[i])
            # 合拢目标 = 离默认（张开）角更远的那个限位端
            close_to = hi if abs(hi - d) > abs(d - lo) else lo
            closed.append(round(close_to, 3))
            print(f"  [{i:2d}] {n:28s} lo={lo:+.3f} hi={hi:+.3f} open={d:+.3f} close={close_to:+.3f}")
        # 输出整只手的"握紧关节角向量"，写脚本专家时直接复制使用
        print(f"  -> {side} closed-q vector: {closed}")

    # --- 第三部分：所有刚体/连杆名字（选末端执行器和指尖连杆时对照用） ---
    print("\n================ BODY / LINK NAMES ================")
    for i, b in enumerate(robot.data.body_names):
        print(f"[{i:2d}] {b}")

    # --- 第四部分：物体相对左右手腕的位置（判断该用哪只手抓） ---
    def body_pos(link):
        """按连杆名查询它在世界坐标系中的位置；找不到则返回 None。"""
        ids, _ = robot.find_bodies(link)
        return robot.data.body_pos_w[0, ids[0]] if ids else None

    obj_pos = obj.data.root_pos_w[0]  # 物体根刚体的世界坐标
    print("\n================ OBJECT vs WRISTS ================")
    print(f"object world pos: {obj_pos.tolist()}")
    # 分别计算物体到左右手腕（wrist_yaw_link 是腕部最末端连杆）的直线距离
    for side, link in [("left", "left_wrist_yaw_link"), ("right", "right_wrist_yaw_link")]:
        p = body_pos(link)
        if p is not None:
            d = torch.norm(obj_pos - p).item()
            print(f"{side:5s} {link}: pos={p.tolist()}  dist_to_object={d:.3f} m")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
