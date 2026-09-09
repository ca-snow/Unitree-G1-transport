# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
在 pickplace 环境里回放行为克隆策略 —— 管线第 3 步（共 4 步）。

位置：bc_train.py（第 2 步，训练）之后、finetune_from_bc.py（第 4 步，
PPO 微调）之前的"体检"环节。做三件事：
  1. 以确定性方式运行 BC 策略（直接取高斯分布的均值，不采样噪声）；
  2. 用与 grasp-expert.py 完全相同的成功判据统计"放置成功率"；
  3. 可选录制视频（--record_video）供目视验收。
如果克隆后的成功率远低于专家，应该先补采数据 / 加长训练，再考虑微调。

"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a BC policy on the pickplace task.")
# --task：要评估的 gym 任务 id（G1 29 关节 + Dex3 手的抓放任务）。
parser.add_argument("--task", type=str, default="Isaac-PickPlace-RL-G129-Dex3-v0")
# --checkpoint：bc_train.py 输出的 bc_policy.pt 检查点路径（必填）。
parser.add_argument("--checkpoint", type=str, required=True)
# --workpiece：工件类型。必须与该检查点【训练时】的工件一致——评估要在与
# 数据采集完全相同的"物体/出生点/放置目标"设定下进行（见 EVAL_PROFILES），否则分数没有意义。
parser.add_argument(
    "--workpiece", type=str, default="cube", choices=["cube", "bolt", "nut", "drill"])
# --num_envs：并行环境数（32 个环境同时评估，成功率统计更稳）。
parser.add_argument("--num_envs", type=int, default=32)
# --rollouts：重复评估几轮（每轮 reset 一次，总样本 = num_envs * rollouts）。
parser.add_argument("--rollouts", type=int, default=4)
# --horizon：每轮走多少个环境步。725 = 专家完整抓放周期的长度，
# 且小于环境 800 步的超时上限（超时会触发 env 自动 reset，打乱统计）。
parser.add_argument("--horizon", type=int, default=725)
# --record_video：录制评估视频（会自动打开相机渲染）。
parser.add_argument("--record_video", action="store_true")
# --seed：随机种子（None = 不固定）。
parser.add_argument("--seed", type=int, default=None)
# --hold：每 N 个环境步才查询一次策略，中间保持（HOLD）上一个动作不变。
parser.add_argument("--hold", type=int, default=1)
# --dump_obs：把0号环境、第1轮rollout前200步的obs/action对保存到
# 指定 .pt 文件（同时记录每个 obs 索引背后的关节名）。
parser.add_argument("--dump_obs", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 录视频必须开相机渲染
if args_cli.record_video:
    args_cli.enable_cameras = True

# 把剩余参数交给 hydra（Isaac Lab 的配置系统）解析
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

import tasks.rl.g1_pickplace  
from tasks.rl.g1_transport.transport_scene_cfg import make_workpiece_cfg
import isaaclab_tasks
from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab.utils.math import quat_apply

from bc_train import SharedPolicy

# 完整周期的成功判据：物体被放到目标点上、回到桌面高度、且处于静止。
# 三个容差分别是：
PLACE_XY_TOL = 0.04     # 水平面(xy)上离放置目标的距离 < 4 cm
PLACE_Z_TOL = 0.03      # 最终高度与"静置高度+期望高度差"的偏差 < 3 cm
PLACE_SPEED_TOL = 0.10  # 物体线速度 < 0.10 m/s（静止，不是被扔到目标上的）

# 各字段含义：
#   spawn_fwd/right  出生点沿"机器人前方/右方"的偏移量（对应世界系 -y / -x）
#   spawn_jitter     覆盖环境默认 ±3 cm 的物体重置噪声（None = 用环境默认值）
#   place_box        (right_lo, right_hi, fwd_lo, fwd_hi)：以出生点为中心的
#                    放置目标采样框（None = 保留环境自己的随机目标）
#   place_root_dz    期望最终根(root)高度 - 出生时根高度
#                    （立放的电钻：+0.065 m，站起来后根部高 6.5 cm）
#   upright_axis     结束时必须指向"世界正上方"的物体局部坐标轴
#                    （None = 不检查竖直朝向）
EVAL_PROFILES = {
    "cube": dict(spawn_fwd=0.0, spawn_right=0.0, spawn_jitter=None,
                 place_box=None, place_root_dz=0.0, upright_axis=None),
    # 螺栓要求最终立直（局部 +z 轴朝上）
    "bolt": dict(spawn_fwd=0.0, spawn_right=0.0, spawn_jitter=None,
                 place_box=None, place_root_dz=0.0, upright_axis=(0.0, 0.0, 1.0)),
    # 螺母：出生点前移 20 cm、右移 10 cm，抖动收窄到 ±1.5 cm，
    # 放置目标框在出生点左后方一小块区域
    "nut": dict(spawn_fwd=0.20, spawn_right=0.10, spawn_jitter=0.015,
                place_box=(-0.05, 0.02, -0.07, -0.02), place_root_dz=0.0,
                upright_axis=None),
    "drill": dict(spawn_fwd=0.10, spawn_right=0.0, spawn_jitter=0.015,
                  # round-41 调参结论：放下点在出生点【右侧】4~6 cm
                  # （前后 ±3 cm 抖动），与 grasp-expert 一致（round-38
                  # 确定向右；上限压在 6 cm 是因为"放下"和"撤离侧移"
                  # 共享同一份侧向可达距离预算）
                  place_box=(0.04, 0.06, -0.03, 0.03), place_root_dz=0.065,
                  # 电钻立放后局部 -y 轴朝上
                  upright_axis=(0.0, -1.0, 0.0)),
}

# 脚本化专家的各阶段边界步号，用打点
# 遥测：hand->obj 取的是 right_obj_rel_pos 观测项的范数，能看出克隆的rollout是在哪个阶段开始偏离专家轨迹的
PHASE_CHECKPOINTS = {
    19: "HOLD", 64: "UP", 104: "TRAV", 184: "DESCEND", 244: "INSERT",
    324: "CLOSE", 404: "LIFT", 504: "CARRY", 584: "LOWER",
    634: "RELEASE", 724: "RETREAT",
}
# 观测布局（110 维）：joint_pos 0:29 | joint_vel 29:58 | hand_joints 58:72 |
# left_obj_rel 72:75 | right_obj_rel 75:78 | last_action 78:106 | phase 106 | place_target_rel 107:110。

RIGHT_OBJ_REL_SLICE = slice(75, 78)
# body/hand 观测 gather 背后的"资产原始顺序"索引表，从 mdp/observations.py
# 连同解析出的关节名一起写进 --dump_obs 文件，
# 让 bc_obs_diff.py 能断言 transport 那边的资产把同样的索引映射到同样的关节
BODY_OBS_IDX = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
                2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
                12, 16, 20, 22, 24, 26, 28]
HAND_OBS_IDX = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
DUMP_STEPS = 200   # --dump_obs 只记录前 200 步（足够覆盖分歧发生的早期阶段）


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    #评估主函数：搭环境 -> 加载 BC 检查点 -> 跑 rollouts -> 按成功判据统计。

    #参数 env_cfg / agent_cfg 由 hydra 装饰器根据任务 id 注入（环境配置和
    #skrl 智能体配置；本脚本只用 env_cfg，agent_cfg 仅为满足装饰器签名）。
    
    device = args_cli.device or env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        torch.manual_seed(args_cli.seed)

    # 与 grasp-expert.py 相同的 PhysX 显存余量：默认 64 MB 的 GPU 碰撞栈
    # 在接触密集的网格上会溢出并丢弃接触点，从而污染策略被评估时的物理——所以扩大到 2^27 = 128 MB。
    env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 27

    # --- 与 grasp-expert.py 完全相同的工件装置（见 EVAL_PROFILES 注释） ---
    prof = EVAL_PROFILES[args_cli.workpiece]
    if args_cli.workpiece != "cube":
        # 非默认工件：替换场景里的 object 配置，并按 profile 平移出生点
        wp_cfg = make_workpiece_cfg(args_cli.workpiece, env_cfg.scene.object.prim_path)
        old_pos = env_cfg.scene.object.init_state.pos
        # 机器人初始朝向是 -y（初始化时绕 z 转了 -90 度）：
        # 机器人前方 = 世界 -y，机器人右方 = 世界 -x，所以这里做减法
        wp_cfg.init_state.pos = (old_pos[0] - prof["spawn_right"],
                                 old_pos[1] - prof["spawn_fwd"],
                                 wp_cfg.init_state.pos[2])
        env_cfg.scene.object = wp_cfg
    if prof["spawn_jitter"] is not None:
        # 覆盖环境默认的物体重置位置噪声（z 向不抖，物体贴桌面）
        j = prof["spawn_jitter"]
        env_cfg.events.reset_object.params["pose_range"] = {
            "x": (-j, j), "y": (-j, j), "z": (0.0, 0.0)}
    print(f"[bc-play] workpiece={args_cli.workpiece}")

    # 固定观察相机：世界系视角，正对操作台
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (-3.55, -3.55, 1.25)
    env_cfg.viewer.lookat = (-4.24, -4.02, 0.85)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.record_video else None)
    if args_cli.record_video:
        # 一条视频覆盖全部 rollouts（step 0 触发一次录制，长度=horizon*rollouts）
        video_dir = os.path.join(os.getcwd(), "bc_videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args_cli.horizon * args_cli.rollouts,
            disable_logger=True,
        )
        print(f"[bc-play] recording video to: {video_dir}")

    base_env = env.unwrapped
    obj = base_env.scene["object"]

    robot = base_env.scene["robot"]
    rw_id = robot.find_bodies("right_wrist_yaw_link")[0][0]   # 右腕 yaw 连杆的 body 索引
    lw_id = robot.find_bodies("left_wrist_yaw_link")[0][0]    # 左腕 yaw 连杆的 body 索引
    elb_id = robot.find_joints("right_elbow_joint")[0][0]     # 右肘关节索引

    # ---- 加载 BC 检查点，按存档里的维度重建网络并切到推理模式 ----
    ckpt = torch.load(args_cli.checkpoint, map_location=device)
    policy = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"]).to(device)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"[bc-play] loaded {args_cli.checkpoint} (val MSE {ckpt.get('val_mse', float('nan')):.6f})")

    rest_z = None                      # 物体静置高度（首次 reset 时标定）
    total_success, total_envs = 0, 0   # 全局成功计数 / 评估环境总数

    for r in range(args_cli.rollouts):
        obs_dict, _ = env.reset()
        if rest_z is None:
            # 首次 reset 时标定物体的"静置高度"，后续成功判据以它为基准
            rest_z = obj.data.root_pos_w[:, 2].mean().item()

            lw0 = robot.data.body_pos_w[0, lw_id]
            rw0 = robot.data.body_pos_w[0, rw_id]
            ob0 = obj.data.root_pos_w[0]
            rb0 = robot.data.root_pos_w[0]
            print(f"[bc-play] rig @reset: left_wrist=({lw0[0]:+.3f},{lw0[1]:+.3f},{lw0[2]:+.3f}) "
                  f"right_wrist=({rw0[0]:+.3f},{rw0[1]:+.3f},{rw0[2]:+.3f}) "
                  f"obj=({ob0[0]:+.3f},{ob0[1]:+.3f},{ob0[2]:+.3f}) "
                  f"base=({rb0[0]:+.3f},{rb0[1]:+.3f},{rb0[2]:+.3f})")
        if prof["place_box"] is not None:
            # 出生点附近的放置目标框，采样方式与数据采集（grasp-expert.py）
            r_lo, r_hi, f_lo, f_hi = prof["place_box"]
            n = base_env.num_envs
            # 在框内均匀采样"右向/前向"偏移量
            r_off = r_lo + (r_hi - r_lo) * torch.rand(n, device=base_env.device)
            f_off = f_lo + (f_hi - f_lo) * torch.rand(n, device=base_env.device)
            # 若采样点离出生点不足 3 cm，就沿原方向放大到恰好 3 cm
            d = torch.sqrt(r_off**2 + f_off**2).clamp_min(1e-6)
            scale = (0.03 / d).clamp_min(1.0)
            r_off, f_off = r_off * scale, f_off * scale
            # 机器人右方=世界 -x、前方=世界 -y，故写入时取减号
            base_env._place_target_w[:, 0] = obj.data.root_pos_w[:, 0] - r_off
            base_env._place_target_w[:, 1] = obj.data.root_pos_w[:, 1] - f_off

        place_target_w = base_env._place_target_w.clone()
        # 只在第 1 轮 rollout 记录 dump（如果指定了 --dump_obs）
        dump_rows = [] if (args_cli.dump_obs and r == 0) else None
        action = None
        for t in range(args_cli.horizon):
            # --hold 节流：每 hold 步才查询一次策略，中间沿用上一个动作
            if t % args_cli.hold == 0 or action is None:
                with torch.no_grad():
                    action = policy(obs_dict["policy"].to(device))
            if dump_rows is not None and t < DUMP_STEPS:
                # 记录"策略在第 t 步看到的 obs + 它产生的 action"
                dump_rows.append((t, obs_dict["policy"][0].detach().cpu().clone(),
                                  action[0].detach().cpu().clone()))
            obs_dict, _, _, _, _ = env.step(action)

            if t < 160 and t % 20 == 0:
                rel0 = obs_dict["policy"][0, RIGHT_OBJ_REL_SLICE]
                print(f"[bc-play]   t={t:3d} "
                      f"elb act={robot.data.joint_pos[0, elb_id].item():+.2f} "
                      f"wrist_z={robot.data.body_pos_w[0, rw_id, 2].item():.3f} "
                      f"rel=({rel0[0].item():+.3f},{rel0[1].item():+.3f},{rel0[2].item():+.3f})")
            # 阶段边界打点：hand->obj 平均距离、肘角、腕高、物高、物->目标距离
            if t in PHASE_CHECKPOINTS:
                rel = obs_dict["policy"][:, RIGHT_OBJ_REL_SLICE]
                rel0 = rel[0]
                d_tgt = torch.norm(obj.data.root_pos_w[:, :2] - place_target_w[:, :2], dim=-1)
                print(f"[bc-play]   t={t:3d} after {PHASE_CHECKPOINTS[t]:7s} "
                      f"hand->obj={rel.norm(dim=-1).mean().item():.3f} m "
                      f"rel=({rel0[0].item():+.3f},{rel0[1].item():+.3f},{rel0[2].item():+.3f}) | "
                      f"elb={robot.data.joint_pos[0, elb_id].item():+.2f} "
                      f"wrist_z={robot.data.body_pos_w[0, rw_id, 2].item():.3f} "
                      f"obj_z={obj.data.root_pos_w[:, 2].mean().item():.3f} | "
                      f"obj->target xy={d_tgt.mean().item():.3f} m")
        if dump_rows is not None:
            # 保存 dump 文件（供 bc_obs_diff.py 使用）：
            torch.save({
                "source": "bc_play",                     # 来源标记（区别于 transport_demo 的 dump）
                "workpiece": args_cli.workpiece,         # 工件类型
                "checkpoint": args_cli.checkpoint,       # 用的哪个检查点
                "body_idx": BODY_OBS_IDX,                # 身体关节的 obs gather 索引表
                "hand_idx": HAND_OBS_IDX,                # 手部关节的 obs gather 索引表
                "body_names": [robot.joint_names[i] for i in BODY_OBS_IDX],  # 索引对应的关节名（关节序校验用）
                "hand_names": [robot.joint_names[i] for i in HAND_OBS_IDX],
                "tick": [row[0] for row in dump_rows],                # 每行的步号 t
                "obs": torch.stack([row[1] for row in dump_rows]),    # [200,110] 观测
                "act": torch.stack([row[2] for row in dump_rows]),    # [200,28] 动作
            }, args_cli.dump_obs)
            print(f"[bc-play] dumped {len(dump_rows)} obs/action rows (env 0, rollout 1) "
                  f"-> {args_cli.dump_obs}")
        # ---- rollout 结束，按三重判据（位置/高度/静止）逐环境判定成功 ----
        final_pos = obj.data.root_pos_w
        final_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)
        d_xy = torch.norm(final_pos[:, :2] - place_target_w[:, :2], dim=-1)
        success = (
            (d_xy < PLACE_XY_TOL)                    # 判据 1：水平方向落在目标 4 cm 内
            # 判据 2：最终高度。place_root_dz 用来平移"立放"的期望高度
            # （站立的电钻根部比躺放高约 6.5 cm；不加偏移的话，每一次正确的立放都会被误判为 FAILED）。
            & ((final_pos[:, 2] - rest_z - prof["place_root_dz"]).abs() < PLACE_Z_TOL)
            & (final_speed < PLACE_SPEED_TOL)        # 判据 3：物体已静止
        )
        if prof["upright_axis"] is not None:
            # 附加判据：物体局部 upright_axis 旋转到世界系后，其 z 分量
            # （即与"正上方"的点积）必须 > 0.94，约等于倾斜 < 20 度
            ax = torch.zeros_like(final_pos)
            ax[:, 0], ax[:, 1], ax[:, 2] = prof["upright_axis"]
            up_dot = quat_apply(obj.data.root_quat_w, ax)[:, 2]
            success = success & (up_dot > 0.94)
            print(f"[bc-play]   upright: up-dot(mean)={up_dot.mean().item():.3f} "
                  f"(want > 0.94), {(up_dot > 0.94).sum().item()}/{base_env.num_envs}")
        total_success += success.sum().item()
        total_envs += base_env.num_envs
        print(f"[bc-play] rollout {r + 1}/{args_cli.rollouts}: "
              f"PLACED {success.sum().item()}/{base_env.num_envs} "
              f"| obj->target xy(mean)={d_xy.mean().item():.3f} m")

    print(f"[bc-play] TOTAL: {total_success}/{total_envs} = {total_success / total_envs:.2%}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
