# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
G1 + Dex3（右手）抓取-放置任务的"脚本化专家"（scripted expert）。

纯强化学习（RL）训练抓取时，策略一直在"钻奖励空子"（reward hacking）：
它学会甩动手臂把物体撞飞来骗取奖励，而不是真正抓取。
因此改用"脚本化专家"：用确定性的脚本可靠地完成一整套动作——
抓取、抬起、搬运到随机化的放置目标点、轻放、松手、撤离——
并把过程中记录的 (观测 obs, 动作 action) 数据对保存下来，
供后续的行为克隆（Behaviour Cloning, BC，见 bc_train.py）模仿学习。
本文件就是第 1 步：脚本化专家 + 数据记录器。

多工件支持（v4 版本）：命令行参数 --workpiece {cube,bolt,nut,drill}
（立方体/螺栓/螺母/电钻）同时做两件事：
(1) 把固定底座 pickplace 环境（环境本身不变）里生成的物体换成对应工件；
(2) 选择一套抓取"档案"（PROFILE，见下方 PROFILES 字典）：
    cube/bolt 用已验证的"侧向捏取"（SIDE pinch），
    nut/drill 用"自上而下抓取"（TOP-DOWN，手掌翻转朝下、竖直插入）。
每类工件的调参流程与 cube 的第 1-10 轮相同：
开 2 个环境 + 图形界面运行，读取 PLAN 阶段的 IK 残差和各相位遥测输出，
调整档案里的数字，反复迭代。

"""

import argparse
import math
import sys

# 注意：AppLauncher 必须在导入其他 isaaclab 模块之前创建（Isaac Sim 的要求），
# 所以本文件的导入顺序是"先解析命令行 -> 启动仿真 App -> 再导入其余库"。
from isaaclab.app import AppLauncher

# ---------- 命令行参数定义 ----------
parser = argparse.ArgumentParser(description="Scripted grasp expert for G1 Dex3 (plan-then-execute).")
# --task：gym 任务注册名（固定底座的 G1-29 关节 + Dex3 抓放环境）
parser.add_argument("--task", type=str, default="Isaac-PickPlace-RL-G129-Dex3-v0")
# --workpiece：选择工件类别。既决定往环境里生成哪个物体（红方块会被
# 运输场景资产替换），也决定用 PROFILES 里哪套抓取参数（cube/bolt 侧捏，nut/drill 俯抓）。
parser.add_argument(
    "--workpiece", type=str, default="cube", choices=["cube", "bolt", "nut", "drill"],
    help="Which workpiece class to run: selects BOTH the object spawned into the env (the red cube is "
    "replaced by the transport-scene asset) and the grasp PROFILE (side pinch for cube/bolt, top-down "
    "grasp for nut/drill). See PROFILES below.")
# --num_envs：并行仿真环境数（看视频用小值，采数据用大值如 256）
parser.add_argument("--num_envs", type=int, default=2, help="Parallel envs (small for video, large for data).")
# --rollouts：连续执行多少个完整的抓放回合
parser.add_argument("--rollouts", type=int, default=1, help="Number of grasp rollouts to run.")
# --record_video：把第一个回合录成 mp4（便于远程检查动作）
parser.add_argument("--record_video", action="store_true", help="Record an mp4 of the first rollout.")
# --save_dataset：.npz 数据集路径；只有"成功"环境的 (obs, action) 会被追加进去
parser.add_argument("--save_dataset", type=str, default=None, help="Path to .npz to append successful (obs, action).")
# --action_noise：DART 风格的数据增广噪声。执行的动作 = 干净动作 + 高斯噪声

parser.add_argument(
    "--action_noise", type=float, default=0.0)
# --place_pitch_deg：临时覆盖档案里放下时手掌前倾角（单位：度），用于
# A/B 对比试验（例如电钻 8 度 vs 9 度）而不必改档案；None = 用档案值。
parser.add_argument("--place_pitch_deg", type=float, default=None)
# --seed：随机种子（复现实验用）
parser.add_argument("--seed", type=int, default=None)
# 把 Isaac Lab 标准的启动参数（--headless、--device、--enable_cameras 等）挂到解析器上
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 在无显示器（headless）服务器上录视频需要打开渲染管线（相机）。
if args_cli.record_video:
    args_cli.enable_cameras = True

# 剩余未识别的参数交给 hydra（Isaac Lab 的配置系统）处理
sys.argv = [sys.argv[0]] + hydra_args

# 启动 Isaac Sim 应用（必须先于其余 isaaclab 导入）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ======================================================================
# 配置区 —— 边看录制的视频边调这些数
# ======================================================================
EE_LINK = "right_hand_palm_link"        # IK 控制的刚体（右手手掌 link）
# 真正的控制点是"抓取中心"（三个指尖 link 的均值位置），而不是手掌原点，手掌位于指尖后方约 0.11 m
FINGERTIP_LINKS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]

# 控制点 = "捏合中心"（PINCH CENTRE）：闭合状态下拇指指尖与
# 食指/中指指尖中点之间的中点，在规划阶段用运动学方式测得

# ---------------------------------------------------------------------------
# 工件档案 PROFILES（--workpiece 选择其中一套；同时也决定环境里生成哪个物体）
# ---------------------------------------------------------------------------

PROFILES = {
    # 已验证的 6 cm 红色立方体（root 在几何中心）。第 10 轮定稿的数字，未再动过。
    "cube": dict(
        # 侧捏；grasp_right=0.02：捏合中心瞄准 root 右侧 2 cm（微偏补偿）；
        # hover=0.12：下降前在 DESC 点上方 12 cm 悬停。
        approach="side", grasp_right=0.02, grasp_fwd=0.0, grasp_up=0.0, hover=0.12,
        # obj_half=0.03：6 cm 方块半宽 3 cm；thumb_margin=0.01：拇指额外留 1 cm 侧向余量。
        obj_half=0.03, thumb_margin=0.01, insert_clear=0.0,
        # close_fraction=0.45：闭合到全闭目标的 45%（0.9 会穿模，0.45 恰好夹住 5-6 cm 跨度）。
        close_fraction=0.45, preclose=0.0,
        # arm_range：肩pitch±0.4 / 肩roll±0.25 / 肩yaw±0.2（防跨身体分支），
        # 肘 ±6.28 即完全自由（抬升靠它），腕roll±0.25 / 腕pitch±1.2 / 腕yaw±0.4
        # （腕 roll/yaw 收紧 = 手掌朝向被动锁定）。
        arm_range=(0.4, 0.25, 0.2, 6.28, 0.25, 1.2, 0.4),
        # ik_ori_weight=0.15：姿态权重低，位置优先；drop_clearance=0.005：放下时
        # 离桌面留 5 mm 再松手（轻放，不压进桌面）；立方体无需竖立检查。
        ik_ori_weight=0.15, drop_clearance=0.005, upright_check=False,
    ),
    # M20 螺栓 x2.5 缩放，头朝下立着：对 50 mm 直径的螺杆做侧握
    
    "bolt": dict(
        approach="side", grasp_right=0.010, grasp_fwd=0.0, grasp_up=0.105, hover=0.09,
        # obj_half=0.025：螺杆半径 2.5 cm；close_fraction=0.50：比 cube 稍深，
        # 因为圆柱面接触面积小。
        obj_half=0.025, thumb_margin=0.01, insert_clear=0.0,
        close_fraction=0.50, preclose=0.0,
        # 关节窗口与 cube 完全相同（同一侧捏动作族）。
        arm_range=(0.4, 0.25, 0.2, 6.28, 0.25, 1.2, 0.4),
        # upright_check=True：螺栓 root 在底部，翻倒后原点高度几乎不变，
        # 必须靠"轴朝上"检查才能识别倒下的螺栓。
        ik_ori_weight=0.15, drop_clearance=0.005, upright_check=True,
    ),
    # M20 螺母，"非均匀"缩放（xy 方向 x5，z 方向 x2.5）：孔径约 90 mm
    
    "nut": dict(
        approach="top", grasp_right=0.0, grasp_fwd=-0.09, grasp_up=0.10, hover=0.10,
        # obj_half=0.1125：外缘半径（拇指避让用）；insert_clear=0.06：
        # DESC 点在抓取点上方 6 cm，保证预卷曲指尖起始不碰螺母。
        obj_half=0.1125, thumb_margin=0.01, insert_clear=0.06,
        
        close_fraction=0.37, preclose=0.5, spawn_fwd=0.20, spawn_right=0.10,
        place_box=(-0.05, 0.02, -0.07, -0.02), spawn_jitter=0.015,
        arm_range=(1.0, 0.25, 0.25, 6.28, 1.9, 1.0, 1.2),
        # ik_ori_weight=0.6：俯抓时"掌心朝下"是主动目标，必须高权重。
        ik_ori_weight=0.6, drop_clearance=0.005, upright_check=False,
    ),
    # YCB 电钻 x1.0，"立起来放"任务
    "drill": dict(
        
        approach="top", grasp_pitch=0.262,
        # grasp_up=0.02：捏合位在平躺电钻顶面下方；
        # hover=0.10：悬停高度 10 cm。
        grasp_right=0.0, grasp_fwd=0.0, grasp_up=0.02, hover=0.10,
        # obj_half=0.03：握把半厚约 3 cm；insert_clear=0.08：DESC 点在抓取点上方 8 cm
        obj_half=0.03, thumb_margin=0.01, insert_clear=0.08,
        # spawn_right=0
        close_fraction=0.45, preclose=0.0, spawn_fwd=0.10, spawn_right=0.0,
        # place_box：电钻专用，放下点在出生点"右侧" 4-6 cm
        place_box=(0.04, 0.06, -0.03, 0.03), spawn_jitter=0.015,
        # 电钻专用关节窗口
        arm_range=(1.3, 0.25, 0.25, 6.28, 1.9, 1.0, 1.55),
        ik_ori_weight=0.6, drop_clearance=0.005,
        # place_reorient=True：立起来放；place_root_dz=0.065：站立 root
        # 比平躺高 6.5 cm。
        place_reorient=True, place_root_dz=0.065,
       
        place_pitch=0.157, place_thumb_up=0.0, retreat_side=0.08,
        place_shove_comp=0.02,
        upright_check=True, upright_axis=(0.0, -1.0, 0.0),
    ),
}
# ---------- 把选中的档案展开成全局常量 ----------
_P = PROFILES[args_cli.workpiece]
APPROACH = _P["approach"]
GRASP_PITCH = _P.get("grasp_pitch", 0.0)  # 倾斜俯抓：手指指向前下方的角度（弧度）
PINCH_OFFSET_RIGHT = _P["grasp_right"]  # 正值 = 瞄准物体 root 右侧（机器人右向）
PINCH_OFFSET_FWD = _P["grasp_fwd"]      # 正值 = 瞄准物体 root 前方（机器人前向）
PINCH_OFFSET_UP = _P["grasp_up"]        # 正值 = 瞄准物体 root 上方
HOVER_HEIGHT = _P["hover"]              # 下降前在 DESC 点上方悬停的高度
OBJ_HALF = _P["obj_half"]               # 抓取部位处的物体半宽（拇指避让用）
THUMB_MARGIN = _P["thumb_margin"]       # 越过物体侧面后额外的侧向净空（侧抓用）
INSERT_CLEAR = _P["insert_clear"]       # DESC 点在抓取点上方的竖直距离（俯抓用）
CLOSE_FRACTION = _P["close_fraction"]   # 手指闭合比例（相对全闭目标）
LIFT_TIGHTEN = _P.get("lift_tighten", 0.0)  # 抬升期间额外渐进加深的闭合量（防滑；孔抓取禁用）
PRECLOSE = _P["preclose"]               # 进近阶段食指/中指的预闭合比例（穿孔用）
SPAWN_FWD = _P.get("spawn_fwd", 0.0)    # 出生点沿机器人前向的平移（大件放得更深）
SPAWN_RIGHT = _P.get("spawn_right", 0.0)  # 出生点沿机器人右向的平移（进入右臂舒适区）
PLACE_BOX = _P.get("place_box", None)   # 出生点周围的放置采样盒 (right_lo, right_hi, fwd_lo, fwd_hi)
SPAWN_JITTER = _P.get("spawn_jitter", None)  # 覆盖环境默认 ±3 cm 的物体重置噪声
ARM_RANGE = _P["arm_range"]             # IK 求解的每关节窗口（围绕初始角的半宽）
PLACE_DROP_CLEARANCE = _P["drop_clearance"]  # 松手时物体离桌面的高度（轻放余量）
UPRIGHT_CHECK = _P["upright_check"]     # 成功判定是否额外检查"竖直朝上"
UPRIGHT_AXIS = _P.get("upright_axis", (0.0, 0.0, 1.0))  # 必须指向世界上方的"物体局部轴"
PLACE_REORIENT = _P.get("place_reorient", False)  # 立起来放：CARRY 期间手腕反滚转
PLACE_ROOT_DZ = _P.get("place_root_dz", 0.0)      # 放好后 root 高度减去出生 root 高度
PLACE_PITCH = _P.get("place_pitch", 0.0)          # 放下时腕 pitch 差量：指尖向前下方倾
if args_cli.place_pitch_deg is not None:          # 命令行 A/B 覆盖（见 --place_pitch_deg）
    PLACE_PITCH = math.radians(args_cli.place_pitch_deg)
PLACE_THUMB_UP = _P.get("place_thumb_up", 0.0)    # 放下时腕 roll 差量：拇指侧高于食指侧
RETREAT_SIDE = _P.get("retreat_side", 0.0)        # 松手后、上升前，手掌向机器人右侧的横移量
PLACE_SHOVE_COMP = _P.get("place_shove_comp", 0.0)  # 按横移实测的右向拖拽量把放下点预先"左偏"
# 注意：LIFT（抬升）不是单独求解的 IK 位姿：手指闭合后，手臂只是沿关节
# 空间把下降路径原样倒放回悬停路点（q_grasp -> q_hover）。

# 手掌姿态目标。侧抓（SIDE）：直接用"初始（rest）姿态"，不做任何改动，
# 并配低求解权重——位置先收敛，手掌朝向由收紧的腕关节窗口被动维持

GRASP_ROLL = 0.0

# 右手手指"完全闭合"目标角（弧度），按动作顺序
# [thumb0, thumb1, thumb2, middle0, middle1, index0, index1]（拇指0/1/2、中指0/1、食指0/1）。
# thumb0 是拇指的外展/旋转关节；+1.047 和 -1.047都会把拇指向"外"叉开，让 thumb1/thumb2 的屈曲来卷曲拇指。
# 实际下发的闭合量 = RIGHT_FINGER_CLOSED * CLOSE_FRACTION（按档案）；
RIGHT_FINGER_CLOSED = [0.0, -1.047, -1.745, 1.571, 1.745, 1.571, 1.745]

# 相位边界，单位是控制步（100 Hz）。总时长 = HORIZON。
# 关键约束：HORIZON 必须"小于"环境的回合超时
# （episode_length_s=8.0 -> 800 控制步）：完整循环：抓取 + 抬升 + 搬运到随机放置目标 + 轻放 +松手 + 撤离。

T_HOLD = 20        # 手臂冻结在初始姿态；物体沉降；末尾求解各路点
T_UP = 65          # 插值 rest -> q_up（在初始 xy 处竖直上抬）
T_TRAV = 105       # 插值 q_up -> q_hover（水平移动到 DESC 点上方，掌心翻转）
T_TILT = 135       # （仅 grasp_pitch）手掌在悬停高度转到倾斜进近姿态
T_DESCEND = 185    # 插值到 q_desc（竖直下降到进近停靠点）
T_INSERT = 245     # 慢速插值 q_desc -> q_grasp（侧抓：左滑 | 俯抓：下压）
T_SETTLE = 255     # 保持 q_grasp；等物体完全静止
T_CLOSE = 325      # 手指逐渐合拢；手臂冻结
T_LIFT = 405       # 沿原路平滑回溯 q_grasp -> q_hover，手指保持闭合
T_HOLD2 = 425      # 悬停保持（物体在手）；末尾求解"放置段"路点
T_TRAV2 = 505      # CARRY：q_hover -> q_phover（移动到放置目标上方）
T_LOWER = 585      # 轻柔下降 q_phover -> q_pdown（物体贴近桌面）
T_RELEASE = 635    # 手指逐渐张开；手臂冻结在 q_pdown
T_SIDE = 660       # （仅 retreat_side）手掌在放下高度向机器人右侧横移
T_RETREAT = 695    # 升回高处（若配置了横移则从横移点升起），手指张开
HORIZON = 725      # 冻结在撤离顶点；远早于 800 步超时结束

# 放置成功的容差（PLACE_DROP_CLEARANCE——捏合中心在手指张开前停在
# 拾取抓取高度之上多少，即约 5 mm 的轻放、不往桌面里压是按档案设置的）。

PLACE_XY_TOL = 0.04     # 最终 |物体xy - 目标xy| 的成功阈值（与 object_placed 奖励一致）
PLACE_Z_TOL = 0.03      # 最终 |物体z - 初始z|（回到桌面上，而不是搭在边上）
PLACE_SPEED_TOL = 0.10  # 最终物体速度（轻放，不能还在滚动）

# 离线（纯运动学）IK 求解器参数——只在 PLAN 阶段运行，运动中绝不运行。
IK_ITERS = 400                          # 每个路点的最大 DLS 迭代次数
IK_DAMPING = 0.05                       # DLS 阻尼系数 lambda（运动学求解可以用低阻尼）
IK_STEP = 0.05                          # 每次迭代允许的最大关节变化（弧度）
# 逐关节的"零空间"回拉增益（朝初始姿态）：肩部强（让上臂保持不动），
# 肘/腕弱（真正的运动由它们完成）。
IK_NULL_GAIN = (0.3, 0.5, 0.5, 0.02, 0.02, 0.02, 0.02)  # [肩pitch, 肩roll, 肩yaw, 肘, 腕roll, 腕pitch, 腕yaw]
# 姿态权重按档案。
# 侧抓：低（0.15）——位置必须赢，手掌朝向由腕关节窗口被动维持（0.5 会让求解器用"几厘米的位置"去换"不可达的几度姿态"）。
# 俯抓：高（0.6）——把手掌俯下去本身就是任务，加宽的腕关节窗口使它可达。
IK_ORI_WEIGHT = _P["ik_ori_weight"]
IK_POS_TOL = 0.005                      # 提前终止的位置容差（米）
IK_ANG_TOL_DEG = 5.0                    # 提前终止的姿态容差（度）

# （成功 = 放到目标上；见上方 PLACE_XY_TOL / PLACE_Z_TOL / PLACE_SPEED_TOL）
# ======================================================================

import math
import os

# 项目根目录：加进 sys.path 才能 import 项目内的 tasks 包；
# 环境变量 PROJECT_ROOT 供场景配置内部解析资产路径。
PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import gymnasium as gym

# Isaac Lab 的四元数/坐标变换工具：
#   axis_angle_from_quat      四元数 -> 轴角向量（方向 = 转轴，模长 = 角度）
#   quat_apply                用四元数旋转一个向量
#   quat_conjugate            四元数共轭（= 逆旋转，单位四元数时）
#   quat_from_angle_axis      (角度, 轴) -> 四元数
#   quat_mul                  四元数乘法（旋转的复合，左乘 = 后施加）
#   subtract_frame_transforms 求一个位姿在另一个坐标系下的表达（相对变换）
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

import tasks.rl.g1_pickplace  # noqa: F401  （导入即注册 gym 任务）
from tasks.rl.g1_transport.transport_scene_cfg import make_workpiece_cfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


# 动作向量的关节顺序（必须与 pickplace_env_cfg.py 里 ActionsCfg.joint_names 一致）。
# 前 14 个是手臂（先左后右，各 7 关节），后 14 个是手指（先左后右，各 7 关节）。
ACTION_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]
# 右臂 7 个关节（IK 只控制它们），顺序与 ARM_RANGE / IK_NULL_GAIN 一一对应。
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
# 右手 7 个手指关节，顺序与 RIGHT_FINGER_CLOSED 一一对应。
RIGHT_FINGER_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
]
ACTION_SCALE = 0.5  # 必须与 JointPositionActionCfg(scale=...) 保持一致


def ease(p: float) -> float:
    #Smoothstep 平滑缓动函数：输入进度 p ∈ [0,1]，输出 3p²-2p³。

    #两端导数为 0，即每段插值的起点和终点速度都是零——手臂加速/减速平滑，不会在相位切换处产生速度突变。输入会先被截断到 [0,1]。
    
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


if args_cli.seed is not None:
    torch.manual_seed(args_cli.seed)


@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    #脚本主体：搭环境、逐回合执行相位状态机、记录并保存数据。

    #由 hydra 装饰器注入两个配置对象：
    #env_cfg环境配置（场景、物理、事件、观测/动作空间定义）
    #agent_cfg skrl智能体配置（本脚本不训练，仅因任务入口需要而接收）
    
    device = args_cli.device or env_cfg.sim.device
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # 256 环境采集螺栓数据时，CLOSE 阶段会撑爆 PhysX 默认 64 MB 的 GPU
    # 碰撞栈（大量手指 x 细密螺纹网格；日志显示峰值需求约 99 MB），
    # 并且会"丢弃接触"——手指瞬间不再与工件碰撞，在仿真里表现为
    # 穿透/弹飞。2**27 = 128 MB，带余量地覆盖实测峰值。
    env_cfg.sim.physx.gpu_collision_stack_size = 2 ** 27

    # --- 工件替换：把红方块换成选中的工件类别 ---
    # 调参台架仍是"同一个"固定底座 pickplace 环境（桌子、机器人底座、事件、观测布局全部不变）；只有物体 prim 不同。
    # xy 保持"这张桌子"上验证过的拾取点；z 来自各类别的配置
    # （资产原点偏移各不相同：螺栓 root 在底部，螺母 root 在几何体下方 0.05——两张桌面高度都是 0.794）。
    if args_cli.workpiece != "cube":
        wp_cfg = make_workpiece_cfg(args_cli.workpiece, env_cfg.scene.object.prim_path)
        old_pos = env_cfg.scene.object.init_state.pos
        # spawn_fwd 把工件往桌子深处推，spawn_right 往右臂方向推。
        # pickplace 的机器人面朝 -y（init_rot = Rz(-90)），因此
        # 机器人前向 = 世界 -y，机器人右向 = 世界 -x（所以是减号）。
        wp_cfg.init_state.pos = (old_pos[0] - SPAWN_RIGHT, old_pos[1] - SPAWN_FWD, wp_cfg.init_state.pos[2])
        env_cfg.scene.object = wp_cfg
    if SPAWN_JITTER is not None:
        # 大件工件紧贴手臂摆动区：环境默认 ±3 cm 的重置噪声曾把螺母
        # 直接随机进摆动区（第 19 轮 UP 扫掠碰撞）。收紧噪声、事件
        # 本身不变，BC 仍能看到多样化的起始状态。
        j = SPAWN_JITTER
        env_cfg.events.reset_object.params["pose_range"] = {
            "x": (-j, j), "y": (-j, j), "z": (0.0, 0.0)}
    print(f"[expert] workpiece={args_cli.workpiece} approach={APPROACH} "
          f"(grasp offsets right={PINCH_OFFSET_RIGHT:+.3f} up={PINCH_OFFSET_UP:+.3f}, "
          f"close_fraction={CLOSE_FRACTION})")

    # 把录制相机对准物体/右手（据 inspect-robot.py，物体约在
    # (-4.24, -4.02, 0.84)）。
    env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.eye = (-3.55, -3.55, 1.25)   # 3/4 斜视角（不是正侧面）
    env_cfg.viewer.lookat = (-4.24, -4.02, 0.85)

    # 创建 gym 环境（录像时需要 rgb_array 渲染模式）
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.record_video else None)

    if args_cli.record_video:
        video_dir = os.path.join(os.getcwd(), "expert_videos")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=HORIZON * args_cli.rollouts,  # 所有回合录成一段连续视频
            disable_logger=True,
        )
        print(f"[expert] recording video to: {video_dir}")

    base_env = env.unwrapped                 # 拆掉包装器，拿到底层 Isaac Lab 环境
    robot = base_env.scene["robot"]          # 机器人 articulation 对象
    obj = base_env.scene["object"]           # 工件刚体对象
    num_envs = base_env.num_envs


    act_ids, _ = robot.find_joints(ACTION_JOINT_NAMES)  # 排序后的 id -> 与环境动作项一致
    arm_ids, _ = robot.find_joints(RIGHT_ARM_JOINTS, preserve_order=True)      # 右臂 7 关节（保持书写顺序）
    finger_ids, _ = robot.find_joints(RIGHT_FINGER_JOINTS, preserve_order=True)  # 右手 7 指关节
    ee_id = robot.find_bodies(EE_LINK)[0][0]                  # 手掌 link 的刚体索引
    tip_ids = [robot.find_bodies(n)[0][0] for n in FINGERTIP_LINKS]  # 三个指尖 link 索引
    # 固定底座 articulation 的雅可比矩阵不包含（静止的）基座 link，索引要减 1。
    ee_jacobi_idx = ee_id - 1 if robot.is_fixed_base else ee_id
    act_ids_t = torch.tensor(act_ids, device=device, dtype=torch.long)

    default_q = robot.data.default_joint_pos  # [N, J]，本任务所有受控关节的默认角都是 0
    # 手指闭合目标向量 = 全闭目标 * 档案闭合比例，形状 [1, 7]
    closed_vec = torch.tensor(RIGHT_FINGER_CLOSED, device=device).unsqueeze(0) * CLOSE_FRACTION  # [1, 7]
    # 抬升期间额外渐进加深的挤压量（见 PROFILES.lift_tighten）
    tighten_vec = torch.tensor(RIGHT_FINGER_CLOSED, device=device).unsqueeze(0) * LIFT_TIGHTEN
    # 进近阶段的"预闭合"姿态（穿孔用，见 PROFILES.preclose）：
    # 食指/中指闭合到其目标的 PRECLOSE 比例，拇指保持"完全张开"
    # （它必须从物体外侧经过；卷曲的拇指会落在物体顶上）。
    preclose_vec = closed_vec * PRECLOSE
    preclose_vec[:, :3] = 0.0  # [thumb0, thumb1, thumb2] 在 CLOSE 之前保持张开

    # 读取手臂关节限位（让运动学求解器保持在合法范围内）；不同版本的
    # Isaac Lab 属性名不同，逐个尝试。
    arm_limits = None
    for attr in ("joint_pos_limits", "soft_joint_pos_limits", "joint_limits"):
        if hasattr(robot.data, attr):
            arm_limits = getattr(robot.data, attr)[:, arm_ids]  # [N, 7, 2]
            break
    if arm_limits is None:
        raise RuntimeError("Could not find joint limits on robot.data")
    arm_lo, arm_hi = arm_limits[..., 0].clone(), arm_limits[..., 1].clone()
    # 把求解器的关节窗口收紧到初始角附近（见 ARM_RANGE）：上臂保持不动、
    # 手腕无法把手掌 roll/yaw 转离初始朝向——这样"跨身体"/"前臂旋前"
    # 之类的坏解分支从一开始就不存在。
    rest_arm = default_q[:, arm_ids]
    for j, rng in enumerate(ARM_RANGE):
        arm_lo[:, j] = torch.maximum(arm_lo[:, j], rest_arm[:, j] - rng)   # 下界收紧到 rest - 窗口
        arm_hi[:, j] = torch.minimum(arm_hi[:, j], rest_arm[:, j] + rng)   # 上界收紧到 rest + 窗口
    null_gain = torch.tensor(IK_NULL_GAIN, device=device).unsqueeze(0)  # [1, 7] 零空间回拉增益

    up_off = torch.tensor([0.0, 0.0, PINCH_OFFSET_UP], device=device).unsqueeze(0)     # 抓取点竖直偏移
    hover_vec = torch.tensor([0.0, 0.0, HOVER_HEIGHT], device=device).unsqueeze(0)     # 悬停高度向量

    # 手掌姿态目标（世界坐标系；每个回合物体沉降后构建一次）。
    # rest_quat_w = 初始姿态，用于 UP 路点（手臂上升时手掌姿态不变）。
    # flip_quat_w = 只做"掌心朝下翻转"的姿态（俯抓用；侧抓时 = rest），
    # 用于 HOVER 路点——手腕在"横移过程中、物体上方"转到它，不在UP 阶段转
    # grasp_quat_w = 翻转 + 可选的 grasp_pitch 倾斜，用于desc/grasp；
    # 设置了 grasp_pitch 时，hover->grasp 的姿态变化以独立的 TILT 步骤在悬停高度执行
    # （第 31 轮用户规范：先翻掌，再倾斜，然后才下降）。
    rest_quat_w = torch.zeros((num_envs, 4), device=device)
    rest_quat_w[:, 0] = 1.0    # 初始化为单位四元数 (w=1, x=y=z=0)
    flip_quat_w = torch.zeros((num_envs, 4), device=device)
    flip_quat_w[:, 0] = 1.0
    grasp_quat_w = torch.zeros((num_envs, 4), device=device)
    grasp_quat_w[:, 0] = 1.0
    fwd_w = torch.zeros((num_envs, 3), device=device)  # 水平的"机器人->物体"方向
    fwd_w[:, 1] = -1.0                                  # 初值：世界 -y（机器人面朝的方向）

    # 指尖均值点在手掌坐标系下的偏移（每回合手指张开时测一次）。
    # 只用于倾斜角遥测。
    grasp_offset_local = torch.zeros((num_envs, 3), device=device)
    # "捏合中心"在手掌坐标系下的偏移（手指"虚拟闭合"时测得，见measure_pinch_offset）。
    # 这才是 IK 的控制点。
    pinch_offset_local = torch.zeros((num_envs, 3), device=device)
    # "张开的拇指指尖"在手掌坐标系下的偏移：用于预测下降过程中张开
    # 拇指的位置，从而计算侧向避让偏移量。
    thumb_open_local = torch.zeros((num_envs, 3), device=device)

    def measure_grasp_offset():
        #手指张开状态下，测量指尖均值点和拇指尖在手掌坐标系的偏移。

        #在每次 reset 后（手指必然张开、手是刚性的）调用一次。结果写入
        #grasp_offset_local（指尖均值，[N,3]，仅遥测用）和thumb_open_local（张开拇指尖，[N,3]，避让计算用）。
        
        palm_pos = robot.data.body_pos_w[:, ee_id]
        palm_quat = robot.data.body_quat_w[:, ee_id]
        tips = torch.stack([robot.data.body_pos_w[:, i] for i in tip_ids], dim=1)  # [N, 3, 3]：3 个指尖的世界坐标
        centre_w = tips.mean(dim=1)
        off, _ = subtract_frame_transforms(palm_pos, palm_quat, centre_w)  # 把均值点变换到手掌坐标系
        grasp_offset_local[:] = off
        off_t, _ = subtract_frame_transforms(palm_pos, palm_quat, robot.data.body_pos_w[:, tip_ids[0]])
        thumb_open_local[:] = off_t  # 张开拇指尖在手掌坐标系的位置

    def palm_w() -> torch.Tensor:
        #手掌中心的世界坐标位置，形状 [N, 3]。
        return robot.data.body_pos_w[:, ee_id]

    def pinch_w() -> torch.Tensor:
        #捏合中心的当前世界坐标位置（控制点），形状 [N, 3]。

        #利用"偏移量在手掌坐标系里是刚性不变的"这一事实：
        #手掌位置 + 手掌姿态旋转后的固定偏移 = 闭合钳口中点现在的位置。
        
        return palm_w() + quat_apply(robot.data.body_quat_w[:, ee_id], pinch_offset_local)

    def measure_pinch_offset():
        #虚拟合拢"手指测出捏合中心在手掌坐标系的偏移（IK 控制点）。

        #步骤：把手指关节角直接写成闭合目标用 get_jacobians() 强制 PhysX 刷新运动学(与 IK求解器同一个技巧);
        #读出闭合钳口的中点在手掌坐标系的位置；最后把所有关节状态原样恢复。
        #结果写入 pinch_offset_local [N,3]。
        
        q_snap = robot.data.joint_pos.clone()      # 快照：当前关节角
        qd_snap = robot.data.joint_vel.clone()     # 快照：当前关节速度
        zero_vel = torch.zeros_like(qd_snap)
        q_work = q_snap.clone()
        q_work[:, finger_ids] = closed_vec         # 手指改成"闭合"角度
        robot.write_joint_state_to_sim(q_work, zero_vel)   # 写入仿真（不 step）
        robot.root_physx_view.get_jacobians()      # 强制刷新正运动学
        tf = robot.root_physx_view.get_link_transforms()
        thumb_tip = tf[:, tip_ids[0], :3]                                  # 拇指尖（thumb_2 link）
        finger_tips = 0.5 * (tf[:, tip_ids[1], :3] + tf[:, tip_ids[2], :3])  # 食指+中指尖的中点
        pinch_centre_w = 0.5 * (thumb_tip + finger_tips)   # 钳口中点 = 拇指尖与双指中点的中点
        palm_pos = tf[:, ee_id, :3]
        palm_quat = tf[:, ee_id, 3:7][:, [3, 0, 1, 2]]     # PhysX 四元数 xyzw -> Isaac Lab wxyz
        off, _ = subtract_frame_transforms(palm_pos, palm_quat, pinch_centre_w)  # 转到手掌坐标系
        pinch_offset_local[:] = off
        robot.write_joint_state_to_sim(q_snap, qd_snap)    # 恢复真实状态
        robot.root_physx_view.get_jacobians()              # 再刷新一次运动学缓存
        print(f"[expert]   pinch-centre offset in palm frame (env0): "
              f"{[round(v, 3) for v in off[0].tolist()]} (|.|={off[0].norm().item():.3f} m)")

    def approach_axis_w() -> torch.Tensor:
        #世界坐标系下"手掌->抓取中心"的单位向量（手的进近轴），[N, 3]。
        axis_local = torch.nn.functional.normalize(grasp_offset_local, dim=-1)
        return quat_apply(robot.data.body_quat_w[:, ee_id], axis_local)   # 局部轴旋转到世界系

    def tilt_deg() -> float:
        #进近轴与水平前向的夹角（度，所有环境的均值）。

        #0 度 = 手指笔直指向物体、与桌面平行（整个运动期望的姿态）。
        #用于遥测输出，观察手掌有没有被扭歪。
        
        cos = (approach_axis_w() * fwd_w).sum(dim=-1).clamp(-1.0, 1.0)   # 点积 = 夹角余弦
        return torch.rad2deg(torch.acos(cos)).mean().item()

    def build_rest_target():
        #构建本次运动的手掌姿态目标（写入 rest/flip/grasp_quat_w）。

        #rest_quat_w：初始姿态本身
        #grasp_quat_w：侧抓 = 与 rest 相同；俯抓 = rest 绕水平前向轴滚转 -90 度：
        #手掌从朝机器人左侧转为朝下，手指保持指向前方, 纯手腕滚转动作，上臂/前臂方向不变
        
        #fwd_w 会被刷新，供侧向几何（right_dir_w）和倾斜遥测使用。
        #GRASP_ROLL（绕水平前向轴）是可选的钳口微调偏移，默认 0。
        
        ee_quat_w = robot.data.body_quat_w[:, ee_id]
        to_obj = obj.data.root_pos_w - robot.data.root_pos_w   # 机器人 -> 物体向量
        to_obj[:, 2] = 0.0                                     # 压平到水平面
        fwd_w[:] = torch.nn.functional.normalize(to_obj, dim=-1)   # 归一化得到前向单位向量
        rest_quat_w[:] = ee_quat_w                             # 记录当前（初始）手掌姿态
        if APPROACH == "top":
            # 掌心朝下的滚转：绕 fwd 轴 -90 度把手掌法向从"机器人左"
            # 转到"竖直向下"（对 +fwd 轴用右手定则）。
            roll_ang = torch.full((num_envs,), -math.pi / 2, device=device)
            # 四元数左乘 = 在世界系中"再施加"这个旋转
            flip_quat_w[:] = quat_mul(quat_from_angle_axis(roll_ang, fwd_w), ee_quat_w)
            grasp_quat_w[:] = flip_quat_w
            if abs(GRASP_PITCH) > 1e-6:
               
                pang = torch.full((num_envs,), -GRASP_PITCH, device=device)
                grasp_quat_w[:] = quat_mul(quat_from_angle_axis(pang, right_dir_w()), flip_quat_w)
        else:
            # 侧抓：翻转姿态和抓取姿态都等于初始姿态（手掌全程不转）
            flip_quat_w[:] = ee_quat_w
            grasp_quat_w[:] = ee_quat_w
        if abs(GRASP_ROLL) > 1e-6:
            # 可选的钳口滚转微调（当前所有档案均为 0）
            roll = torch.full((num_envs,), GRASP_ROLL, device=device)
            grasp_quat_w[:] = quat_mul(quat_from_angle_axis(roll, fwd_w), grasp_quat_w)

    def right_dir_w() -> torch.Tensor:
        #指向"机器人右侧"的水平单位向量（= fwd 叉乘 z-up 的展开式），[N, 3]。
        r = torch.stack([fwd_w[:, 1], -fwd_w[:, 0], torch.zeros_like(fwd_w[:, 0])], dim=-1)
        return torch.nn.functional.normalize(r, dim=-1)

    def grasp_point_w(obj_p: torch.Tensor) -> torch.Tensor:
        #捏合中心必须到达的目标点，[N, 3]。
        #= 物体 root 位置 + 档案里的抓取偏移（右向/前向/上向）。
        
        return obj_p + right_dir_w() * PINCH_OFFSET_RIGHT + fwd_w * PINCH_OFFSET_FWD + up_off

    def thumb_clear_shift(gp: torch.Tensor) -> torch.Tensor:
        #计算 DESCEND 路点需要的侧向（机器人右向）偏移量，[N, 1]。
        #仅侧抓（SIDE）使用。

        #预测"张开的拇指尖"在最终抓取位姿下的位置(手掌处于目标姿态、捏合中心在 gp),
        #要求它至少在物体轴线右侧OBJ_HALF + THUMB_MARGIN 处。差多少，下降路点就往右偏多少，
        #之后的 INSERT 相位再水平补回这段距离。
        
        palm_t = gp - quat_apply(grasp_quat_w, pinch_offset_local)   # 由捏合目标反推手掌位置
        thumb_t = palm_t + quat_apply(grasp_quat_w, thumb_open_local)  # 预测张开拇指尖位置
        rel = ((thumb_t - gp) * right_dir_w()).sum(dim=-1, keepdim=True)  # 拇指在目标右侧多远
        return torch.clamp((OBJ_HALF + THUMB_MARGIN) - rel, min=0.0)  # 缺多少偏多少（不为负）

    def desc_point_w(gp: torch.Tensor) -> torch.Tensor:
        #按档案计算 DESC 路点位置，[N, 3]。

        #侧抓（SIDE）：在物体"旁边"（带向机器人右侧的拇指避让偏移；
        #之后 INSERT 向左滑入）。俯抓（TOP）：在抓取点"正上方"
        #INSERT_CLEAR 处（指尖起始时不碰物体；之后 INSERT 竖直下压）。
        
        if APPROACH == "top":
            d = gp.clone()
            d[:, 2] += INSERT_CLEAR    # 俯抓：抬高一个插入净空
            return d
        shift = thumb_clear_shift(gp)
        print(f"[expert]   PLAN shift : thumb side-shift(env0)={shift[0].item():.3f} m")
        return gp + right_dir_w() * shift   # 侧抓：向右偏移避让拇指

    # ------------------------------------------------------------------
    # 运动学正解（FK）+ 离线 IK 求解器（只在 PLAN 阶段使用）
    # ------------------------------------------------------------------
    def fk_palm_pose():
        #从"当前"关节状态计算手掌位姿 + 手臂雅可比矩阵，绕过缓存。

        #get_jacobians() 会让 PhysX 在内部执行 updateArticulationKinematics

        #返回：palm_pos [N,3]、palm_quat [N,4]（wxyz）、jac [N,6,7]（手掌 6 维空间速度对右臂 7 关节的雅可比）。
        
        jac_full = robot.root_physx_view.get_jacobians()
        tf = robot.root_physx_view.get_link_transforms()
        palm_pos = tf[:, ee_id, :3]
        palm_quat = tf[:, ee_id, 3:7][:, [3, 0, 1, 2]]  # PhysX xyzw -> Isaac Lab wxyz
        jac = jac_full[:, ee_jacobi_idx, :, arm_ids]     # 只取右臂 7 列 -> [N, 6, 7]
        return palm_pos, palm_quat, jac

    eye6 = torch.eye(6, device=device).unsqueeze(0)          # 6x6 单位阵（DLS 阻尼项用）
    eye7 = torch.eye(len(arm_ids), device=device).unsqueeze(0)  # 7x7 单位阵（零空间投影用）
    ang_tol = math.radians(IK_ANG_TOL_DEG)

    def solve_arm_ik(pinch_target_w: torch.Tensor, q_start: torch.Tensor, label: str,
                     quat_w: torch.Tensor = None, ori_weight: float = None) -> torch.Tensor:
        #求解右臂关节角，使"捏合中心"到达 pinch_target_w、姿态接近 quat_w。

        #算法：阻尼最小二乘（DLS）迭代 IK + 加权零空间回拉。纯运动学：
        #每次迭代把候选关节角写进仿真求 FK，结束后完整恢复真实动力学状态——物理从不前进，屏幕上什么都不动。

        #输入：pinch_target_w [N,3] 捏合中心目标；q_start [N,7] 迭代初值（用上一个路点的解做"热启动"，保证解的连续性）;
        #label遥测标签；quat_w [N,4] 姿态目标（默认 grasp_quat_w）；ori_weight 姿态权重（默认 IK_ORI_WEIGHT）。
        #输出：q_arm [N,7] 右臂关节角解。
        
        target_quat = grasp_quat_w if quat_w is None else quat_w
        ori_w = IK_ORI_WEIGHT if ori_weight is None else ori_weight
        q_snap = robot.data.joint_pos.clone()    # 快照，最后恢复
        qd_snap = robot.data.joint_vel.clone()
        zero_vel = torch.zeros_like(qd_snap)
        # 把"捏合中心目标"换算成"手掌目标"：偏移量在手掌坐标系是
        # 刚性的，用"目标姿态"旋转它再从目标位置里减掉即可。
        palm_target_w = pinch_target_w - quat_apply(target_quat, pinch_offset_local)
        q_rest_arm = default_q[:, arm_ids]
        q_arm = q_start.clone()
        q_work = q_snap.clone()
        for _ in range(IK_ITERS):
            q_work[:, arm_ids] = q_arm
            robot.write_joint_state_to_sim(q_work, zero_vel)   # 写入候选关节角
            palm_pos, palm_quat, jac = fk_palm_pose()          # FK：算出手掌位姿和雅可比
            pos_err = palm_target_w - palm_pos                 # 位置误差 [N,3]
            # 姿态误差：目标四元数 * 当前四元数的共轭 = "还差的旋转"，
            # 转成轴角向量（方向 = 转轴，模长 = 角度）
            ang_err = axis_angle_from_quat(quat_mul(target_quat, quat_conjugate(palm_quat)))
            if pos_err.norm(dim=-1).max() < IK_POS_TOL and ang_err.norm(dim=-1).max() < ang_tol:
                break   # 所有环境都在容差内 -> 提前收敛
            # 6 维任务误差 = [位置误差, 姿态权重*姿态误差]
            err = torch.cat([pos_err, ori_w * ang_err], dim=-1).unsqueeze(-1)  # [N, 6, 1]
            # DLS 核心：dq = J^T (J J^T + lambda^2 I)^-1 * err
            # （阻尼项防止在奇异位形附近解爆炸）
            jjt = jac @ jac.transpose(1, 2) + (IK_DAMPING ** 2) * eye6                 # [N, 6, 6]
            dq = (jac.transpose(1, 2) @ torch.linalg.solve(jjt, err)).squeeze(-1)      # [N, 7]
            # 加权零空间姿态偏置（朝初始姿态回拉）：投到雅可比零空间里，
            # 不影响末端任务。肩部增益大 = 上臂被"钉"在原位；
            # 肘/腕增益小 = 真正的运动由它们自由完成。
            jpinv = jac.transpose(1, 2) @ torch.linalg.inv(jjt)                        # [N, 7, 6] 伪逆
            null_proj = eye7 - jpinv @ jac                                             # [N, 7, 7] 零空间投影
            dq = dq + (null_proj @ (null_gain * (q_rest_arm - q_arm)).unsqueeze(-1)).squeeze(-1)
            # 步长限幅（每次最多 IK_STEP 弧度）+ 关节窗口截断（arm_lo/hi）
            q_arm = torch.clamp(q_arm + torch.clamp(dq, -IK_STEP, IK_STEP), arm_lo, arm_hi)
        # 最终残差遥测（在解处评估，恢复状态之前）——调参时看的就是这几个数
        q_work[:, arm_ids] = q_arm
        robot.write_joint_state_to_sim(q_work, zero_vel)
        palm_pos, palm_quat, _ = fk_palm_pose()
        pinch_pos = palm_pos + quat_apply(palm_quat, pinch_offset_local)   # 解处的捏合中心位置
        res_vec = pinch_target_w - pinch_pos
        res_p = torch.norm(res_vec, dim=-1)          # 位置残差（米）
        res_a = torch.norm(axis_angle_from_quat(quat_mul(target_quat, quat_conjugate(palm_quat))), dim=-1)  # 角残差
        print(f"[expert]   PLAN {label:6s}: residual pos(max)={res_p.max().item():.4f} m "
              f"xyz(env0)={[round(v, 3) for v in res_vec[0].tolist()]} | "
              f"ang(max)={torch.rad2deg(res_a).max().item():.1f} deg | "
              f"q_arm(env0)={[round(v, 3) for v in q_arm[0].tolist()]}")
        # 恢复真实动力学状态并重新同步运动学缓存
        robot.write_joint_state_to_sim(q_snap, qd_snap)
        robot.root_physx_view.get_jacobians()
        return q_arm

    def report(tag: str, obj_start_xy=None):
        #拾取段的相位边界遥测：打印捏合中心/拇指/手指与物体的相对几何。

        #调参时靠这行输出判断抓取在"哪一步"失败。obj_start_xy 若给出，额外打印物体的水平漂移量（漂移=手把物体撞跑了）。
        
        o = obj.data.root_pos_w
        tgt = grasp_point_w(o)                        # 捏合中心的瞄准点
        dp = torch.norm(pinch_w() - tgt, dim=-1)      # 捏合中心到目标的三维距离
        exy = torch.norm(pinch_w()[:, :2] - tgt[:, :2], dim=-1)   # 水平距离
        # 拇指遥测：到物体轴线的水平距离 + 相对物体 root 的高度。
        # 若物体被拇指"压住"，表现为拇指 xy 落进物体投影内、dz 接近顶面高度。
        th = robot.data.body_pos_w[:, tip_ids[0]]
        t_xy = torch.norm(th[:, :2] - o[:, :2], dim=-1)
        t_dz = th[:, 2] - o[:, 2]
        # 手指遥测（食指/中指尖均值）：掌心朝下的"孔抓取"要求它们落在
        # 孔"内"（xy x7.5 的螺母：距轴线 xy < 0.06），同时拇指留在外缘"外"（xy > 0.113）。
        fg = 0.5 * (robot.data.body_pos_w[:, tip_ids[1]] + robot.data.body_pos_w[:, tip_ids[2]])
        f_xy = torch.norm(fg[:, :2] - o[:, :2], dim=-1)
        f_dz = fg[:, 2] - o[:, 2]
        msg = (f"[expert]   {tag:11s} pinch->obj(mean)={dp.mean().item():.3f} m "
               f"(xy={exy.mean().item():.3f}) | thumb xy={t_xy.mean().item():.3f} "
               f"dz={t_dz.mean().item():+.3f} | fingers xy={f_xy.mean().item():.3f} "
               f"dz={f_dz.mean().item():+.3f} | tilt-vs-fwd={tilt_deg():.0f} deg | "
               f"obj_z(mean)={o[:, 2].mean().item():.3f} m")
        if obj_start_xy is not None:
            drift = torch.norm(o[:, :2] - obj_start_xy, dim=-1)
            msg += f" | obj_xy_drift(mean)={drift.mean().item():.3f} m"
        print(msg)

    def place_report(tag: str):
        #放置段的遥测：物体/捏合中心离放置目标还有多远。
        o = obj.data.root_pos_w
        d_xy = torch.norm(o[:, :2] - place_target_w[:, :2], dim=-1)       # 物体到目标的水平距离
        p_xy = torch.norm(pinch_w()[:, :2] - place_target_w[:, :2], dim=-1)  # 捏合中心到目标的水平距离
        msg = (f"[expert]   {tag:13s} obj->target xy(mean)={d_xy.mean().item():.3f} m | "
               f"pinch->target xy={p_xy.mean().item():.3f} | obj_z(mean)={o[:, 2].mean().item():.3f} m")
        if UPRIGHT_CHECK:
            # 逐相位边界统计"仍竖直"的环境数：定位立放是在"哪一步"
            # 丢掉的（RELEASE 时还站着、RETREAT 时倒了 = 手撤离时
            # 撞倒的；RELEASE 时就倒了 = 放下时就歪了）。
            ax = torch.zeros_like(o)
            ax[:, 0], ax[:, 1], ax[:, 2] = UPRIGHT_AXIS
            ud = quat_apply(obj.data.root_quat_w, ax)[:, 2]   # 局部竖直轴旋到世界系后的 z 分量
            msg += f" | upright {(ud > 0.94).sum().item()}/{num_envs} (up-dot mean {ud.mean().item():.2f})"
        print(msg)

    rest_z = None                              # 物体出生时的桌面高度（第一回合记录）
    all_obs, all_act, all_success = [], [], []  # 跨回合累积的数据缓冲

    # ================== 回合主循环 ==================
    for r in range(args_cli.rollouts):
        obs_dict, _ = env.reset()
        measure_grasp_offset()  # reset 后手指必然张开 -> 手是刚性的，可以测偏移
        measure_pinch_offset()  # 虚拟闭合 -> 捏合中心偏移（控制点）
        rest_d = torch.norm(pinch_w() - obj.data.root_pos_w, dim=-1).mean().item()
        print(f"[expert] rollout {r + 1}: rest-pose pinch->obj(mean)={rest_d:.3f} m")
        obj_start = obj.data.root_pos_w.clone()      # 物体出生位置（放置盒采样基准）
        obj_start_xy = obj_start[:, :2].clone()      # 漂移遥测基准
        if rest_z is None:
            rest_z = obj_start[:, 2].mean().item()   # 记录桌面上的静止高度

        q_rest_arm = default_q[:, arm_ids].clone()
        q_up = q_hover = q_desc = q_grasp = None  # 拾取路点（HOLD 末尾填充）
        q_tilt = None                             # 倾斜进近步骤（仅 grasp_pitch）
        q_phover = q_pdown = None                 # 放置路点（HOLD2 末尾填充）
        q_pside = q_pside_up = None               # retreat_side 路点（仅立放）
        grasp_z = None                            # 捏合中心的抓取高度，放下时复用
        # 每环境随机化的放置目标（世界系），由环境的 reset_place_target事件采样；
        # 同时通过 place_target_rel 观测喂给策略。
        if PLACE_BOX is not None:
           
            r_lo, r_hi, f_lo, f_hi = PLACE_BOX
            r_off = r_lo + (r_hi - r_lo) * torch.rand(num_envs, device=device)   # 右向偏移采样
            f_off = f_lo + (f_hi - f_lo) * torch.rand(num_envs, device=device)   # 前向偏移采样
            # 强制位移 >= 3 cm，保证放置永远不是"原地不动"
            d = torch.sqrt(r_off**2 + f_off**2).clamp_min(1e-6)
            scale = (0.03 / d).clamp_min(1.0)     # 位移不足 3 cm 时按比例放大
            r_off, f_off = r_off * scale, f_off * scale
            # 机器人右向 = 世界 -x，前向 = 世界 -y，所以都是减
            base_env._place_target_w[:, 0] = obj_start[:, 0] - r_off
            base_env._place_target_w[:, 1] = obj_start[:, 1] - f_off
        place_target_w = base_env._place_target_w.clone()
        print(f"[expert]   place target(env0, world)={[round(v, 3) for v in place_target_w[0].tolist()]}")

        # 本回合的数据缓冲：观测 [HORIZON, N, obs维] / 动作 [HORIZON, N, 28]
        roll_obs = torch.zeros((HORIZON, num_envs, obs_dict["policy"].shape[-1]), device=device)
        roll_act = torch.zeros((HORIZON, num_envs, len(ACTION_JOINT_NAMES)), device=device)

        # ---------- 相位状态机：按控制步 t 决定手臂目标 arm_q 和手指指令 ----------
        # 每个分支设置三个手指控制量：
        #   frac主闭合进度（0=张开, 1=闭合到 close_fraction）
        #   pre 预闭合进度（食指/中指的"半合"底线，穿孔用）
        #   tight抬升加压进度（lift_tighten 的激活量）
        for t in range(HORIZON):
            tight = 0.0  # lift_tighten 激活量（仅 LIFT..RELEASE 非零）
            if t < T_HOLD:
                # HOLD：手臂冻结在初始姿态，让（位置随机化的）物体沉降，
                # 最后一步用它的最终位姿规划全部拾取路点。
                arm_q = q_rest_arm
                frac, pre = 0.0, 0.0
                if t == T_HOLD - 1:
                    build_rest_target()                    # 构建姿态目标（rest/flip/grasp）
                    obj_p = obj.data.root_pos_w.clone()
                    gp = grasp_point_w(obj_p)              # 捏合中心的瞄准点
                    grasp_z = gp[:, 2].clone()             # 记下抓取高度，放下时复用
                    desc = desc_point_w(gp)                # 按档案计算进近停靠点
                    # 竖直优先：捏合中心在其"初始 xy"处直着抬高，且用
                    # "初始姿态"（低姿态权重）求解——转到抓取姿态的手腕
                    # 旋转发生在 TRAV（横移）期间、物体上方，绝不在 UP。
                    up_target = pinch_w().clone()
                    up_target[:, 2] = desc[:, 2] + HOVER_HEIGHT
                    q_up = solve_arm_ik(up_target, q_rest_arm, "up", quat_w=rest_quat_w, ori_weight=0.15)
                    # hover 用"纯翻掌"姿态求解；grasp_pitch 的倾斜是同一
                    # 位置上的独立 TILT 步骤
                    q_hover = solve_arm_ik(desc + hover_vec, q_up, "hover", quat_w=flip_quat_w)
                    if abs(GRASP_PITCH) > 1e-6:
                        q_tilt = solve_arm_ik(desc + hover_vec, q_hover, "tilt")   # 同位置、倾斜姿态
                    q_desc = solve_arm_ik(desc, q_tilt if q_tilt is not None else q_hover, "desc")
                    q_grasp = solve_arm_ik(gp, q_desc, "grasp")
            elif t < T_UP:
                # UP：rest -> q_up 平滑插值（竖直上抬）
                s = ease((t - T_HOLD) / (T_UP - T_HOLD))
                arm_q = q_rest_arm + s * (q_up - q_rest_arm)
                frac, pre = 0.0, s  # 上抬的同时，食指/中指渐进到预闭合姿态
            elif t < T_TRAV:
                # TRAV：q_up -> q_hover 平滑插值（水平横移 + 手腕翻掌）
                s = ease((t - T_UP) / (T_TRAV - T_UP))
                arm_q = q_up + s * (q_hover - q_up)
                frac, pre = 0.0, 1.0
            elif t < T_TILT and q_tilt is not None:
                # TILT：悬停高度上、工件正上方的专用步骤——手掌从"平掌
                # 朝下"转到倾斜进近姿态，之后才允许下降
                s = ease((t - T_TRAV) / (T_TILT - T_TRAV))
                arm_q = q_hover + s * (q_tilt - q_hover)
                frac, pre = 0.0, 1.0
            elif t < T_DESCEND:
                # DESCEND：下降到进近停靠点（有 TILT 则从 q_tilt 出发）
                if q_tilt is not None:
                    s = ease((t - T_TILT) / (T_DESCEND - T_TILT))
                    arm_q = q_tilt + s * (q_desc - q_tilt)
                else:
                    s = ease((t - T_TRAV) / (T_DESCEND - T_TRAV))
                    arm_q = q_hover + s * (q_desc - q_hover)
                frac, pre = 0.0, 1.0
            elif t < T_INSERT:
                # INSERT，按档案分两种。侧抓：水平"左"滑（张开的拇指从
                # 物体右侧面外侧横着进入，而不是落在顶面上）。俯抓：
                # 竖直"下"压（预闭合的手指穿进孔里，张开的拇指从墙壁
                # 外侧经过）。
                s = ease((t - T_DESCEND) / (T_INSERT - T_DESCEND))
                arm_q = q_desc + s * (q_grasp - q_desc)
                frac, pre = 0.0, 1.0
            elif t < T_SETTLE:
                # SETTLE：保持抓取位不动，等物体完全静止
                arm_q = q_grasp
                frac, pre = 0.0, 1.0
            elif t < T_CLOSE:
                # CLOSE：手臂冻结，手指线性合拢（frac 从 0 涨到 1）
                arm_q = q_grasp
                frac, pre = (t - T_SETTLE) / max(1, (T_CLOSE - T_SETTLE)), 1.0
            elif t < T_LIFT:
                # LIFT = 沿原路平滑"回溯"下降路径（q_grasp -> q_hover）。
                # 这里"不"新解 IK 位姿：两个端点都在下降途中被干净地执行
                # 过，手臂按原路上升，绝无扭曲解的可能
                s = ease((t - T_CLOSE) / (T_LIFT - T_CLOSE))
                arm_q = q_grasp + s * (q_hover - q_grasp)
                frac, pre, tight = 1.0, 1.0, s  # 工件越升越高，加压随之加深
            elif t < T_HOLD2:
                # HOLD2：物体在手中、悬停保持；末尾规划"放置段"路点
                # （运动学求解，机器人不动）。
                arm_q = q_hover
                frac, pre, tight = 1.0, 1.0, 1.0
                if t == T_HOLD2 - 1:
                    if PLACE_REORIENT:
                        # 立放电钻：放置路点用"初始（rest）姿态"
                        # 求解——CARRY 插值随后把手腕绕机器人前向轴反滚转+90 度，横握的工件在空中旋转为竖直。
                        pp_nom = place_target_w + right_dir_w() * PINCH_OFFSET_UP + fwd_w * PINCH_OFFSET_FWD
                        obj_rel = obj.data.root_pos_w - pinch_w()   # 捏合中心->物体（此时仍横握）
                        c_f = (obj_rel * fwd_w).sum(dim=-1, keepdim=True)   # 手内偏移的前向分量
                        c_u = obj_rel[:, 2].unsqueeze(-1)                    # 手内偏移的竖直分量
                        # 旋转补偿：前向分量原样保留，"上向"分量映射到右向
                        pp = place_target_w - (fwd_w * c_f + right_dir_w() * c_u)
                        pp[:, :2] = pp_nom[:, :2] + (pp - pp_nom)[:, :2].clamp(-0.05, 0.05)  # ±5 cm 截断
                       
                        pp -= right_dir_w() * PLACE_SHOVE_COMP
                        # 放下高度 = 桌面 + 站立增高 + 轻放余量 - 抓取点右偏
                        # （右偏在旋转后变成竖直分量，要扣掉）
                        pp[:, 2] = rest_z + PLACE_ROOT_DZ + PLACE_DROP_CLEARANCE - PINCH_OFFSET_RIGHT
                        # 下面 IK 求解后追加的腕 pitch 差量会让捏合中心下沉
                        # 约 sin(pitch)*|捏合偏移|；预先把放下点抬高同样的量，底座才能仍以轻放余量落桌，而不是压进桌面。
                        pp[:, 2] += math.sin(PLACE_PITCH) * 0.09
                        q_phover = solve_arm_ik(pp + hover_vec, q_hover, "phover", quat_w=rest_quat_w)
                        q_pdown = solve_arm_ik(pp, q_phover, "pdown", quat_w=rest_quat_w)
                        if RETREAT_SIDE > 0.0:
                            # 松手后的撤离路线：先在放下高度向右横移，
                            # "然后"从横移点竖直上升
                            side = right_dir_w() * RETREAT_SIDE
                            q_pside = solve_arm_ik(pp + side, q_pdown, "pside", quat_w=rest_quat_w)
                            q_pside_up = solve_arm_ik(pp + side + hover_vec, q_pside, "pret", quat_w=rest_quat_w)
                        if PLACE_PITCH != 0.0 or PLACE_THUMB_UP != 0.0:
                            # 纯手掌的底座调平：指尖
                            # 向前下方倾（wrist_pitch 取正）、拇指侧抬到
                            # 食指侧之上（wrist_roll 取负），抵消电钻在
                            # 手中固有的倾斜。
                            dq = torch.zeros((1, len(arm_ids)), device=device)
                            dq[0, 4] = -PLACE_THUMB_UP  # 腕 roll：拇指侧抬高
                            dq[0, 5] = PLACE_PITCH      # 腕 pitch：指尖下倾
                            q_pdown = q_pdown + dq
                            if q_pside is not None:
                                q_pside = q_pside + dq
                                q_pside_up = q_pside_up + dq
                    else:
                        # 平放：用"实测"的捏合中心->物体偏移瞄准，而不是
                        # 名义的拾取偏移：闭合+抬升会把螺母在手中拖动
                        # 6-9 cm，按名义偏移瞄准就会偏离目标那么远才松手。
                        nominal = right_dir_w() * PINCH_OFFSET_RIGHT + fwd_w * PINCH_OFFSET_FWD
                        dev = (pinch_w() - obj.data.root_pos_w - nominal)[:, :2].clamp(-0.05, 0.05)  # 实测偏差（截断）
                        pp = place_target_w + nominal
                        pp[:, :2] += dev
                        pp[:, 2] = grasp_z + PLACE_DROP_CLEARANCE   # 放下高度 = 抓取高度 + 轻放余量
                        q_phover = solve_arm_ik(pp + hover_vec, q_hover, "phover")
                        q_pdown = solve_arm_ik(pp, q_phover, "pdown")
            elif t < T_TRAV2:
                # CARRY：关节空间插值到放置目标上方（水平搬运；立放时
                # 这段插值同时完成手腕反滚转、工件在空中转正）。
                s = ease((t - T_HOLD2) / (T_TRAV2 - T_HOLD2))
                arm_q = q_hover + s * (q_phover - q_hover)
                frac, pre, tight = 1.0, 1.0, 1.0
            elif t < T_LOWER:
                # LOWER：轻柔放下——物体最终停在桌面上方约 PLACE_DROP_CLEARANCE 处
                s = ease((t - T_TRAV2) / (T_LOWER - T_TRAV2))
                arm_q = q_phover + s * (q_pdown - q_phover)
                frac, pre, tight = 1.0, 1.0, 1.0
            elif t < T_RELEASE:
                # RELEASE：手指逐渐松开；手臂冻结 -> 物体轻柔落定。
                # 这里手指只张回到"预闭合"姿态：孔抓取时它们还在孔里，完全伸直会把物体拨来拨去。
                arm_q = q_pdown
                frac, pre = 1.0 - (t - T_LOWER) / max(1, (T_RELEASE - T_LOWER)), 1.0
                tight = frac  # 额外挤压随松开进度一起消退
            elif t < T_SIDE and q_pside is not None:
                # SIDE：上升"之前"先在放下高度向机器人右侧横移
                # （立放：电机外壳在松开的握持位上方鼓出；直接竖直上升会让虎口勾到它）。
                s = ease((t - T_RELEASE) / (T_SIDE - T_RELEASE))
                arm_q = q_pdown + s * (q_pside - q_pdown)
                frac, pre = 0.0, 1.0
            elif t < T_RETREAT:
                # RETREAT：升回高处（若配置了横移则从横移点升起）；
                # 手指保持预闭合姿态，直到完全离开物体。
                if q_pside is not None:
                    s = ease((t - T_SIDE) / (T_RETREAT - T_SIDE))
                    arm_q = q_pside + s * (q_pside_up - q_pside)
                else:
                    s = ease((t - T_RELEASE) / (T_RETREAT - T_RELEASE))
                    arm_q = q_pdown + s * (q_phover - q_pdown)
                frac, pre = 0.0, 1.0
            else:
                # 冻结在撤离顶点；预闭合的手指（此时已远高于物体）在
                # 前 20 步内放松到完全张开。
                arm_q = q_phover if q_pside is None else q_pside_up
                frac, pre = 0.0, max(0.0, 1.0 - (t - T_RETREAT) / 20.0)

            # 组装期望的"绝对关节角"，再换算成原始动作向量。
            # 手指指令 = 预闭合底线（仅食指/中指，由 `pre` 控制）+
            # 剩余到全闭的部分（由 `frac` 控制）+ 抬升加压（`tight`）；
            # preclose=0 时严格退化为老的 closed_vec * frac。
            q_des = default_q.clone()
            q_des[:, arm_ids] = arm_q          # 右臂 7 关节 = 状态机给出的目标
            q_des[:, finger_ids] = preclose_vec * pre + (closed_vec - preclose_vec) * frac + tighten_vec * tight
            # 反推动作：action = (目标角 - 默认角) / scale（环境内部会乘回去）
            action = (q_des[:, act_ids_t] - default_q[:, act_ids_t]) / ACTION_SCALE  # [N, 28]

            roll_obs[t] = obs_dict["policy"]   # 记录观测（step 之前的观测，与动作对齐）
            roll_act[t] = action  # 记录的标签 = "干净"动作，即使执行时加了噪声
            exec_action = action
            if args_cli.action_noise > 0.0:
                # DART 式扰动：执行的动作加高斯噪声，标签保持干净
                exec_action = action + args_cli.action_noise * torch.randn_like(action)
            obs_dict, _, _, _, _ = env.step(exec_action)   # 推进仿真一步

            # 相位边界遥测：告诉我们抓取是在"哪一步"失败的
            if t == T_HOLD - 1:
                report("after HOLD", obj_start_xy)     # tilt = 初始姿态与前向的夹角（约 20 度算正常）
            elif t == T_UP - 1:
                report("after UP", obj_start_xy)       # 此时 obj_xy_drift 必须仍约为 0
            elif t == T_TRAV - 1:
                report("after TRAV", obj_start_xy)     # pinch->obj 应约等于 HOVER_HEIGHT
                # 在悬停点"重规划"：物体可能被碰过，用它"当前"的位置
                # 重解下降 + 抓取路点（运动学求解，机器人不动）。
                obj_p = obj.data.root_pos_w.clone()
                gp = grasp_point_w(obj_p)
                grasp_z = gp[:, 2].clone()
                desc = desc_point_w(gp)
                if q_tilt is not None:
                    q_tilt = solve_arm_ik(desc + hover_vec, q_hover, "tilt*")
                q_desc = solve_arm_ik(desc, q_tilt if q_tilt is not None else q_hover, "desc*")
                q_grasp = solve_arm_ik(gp, q_desc, "grasp*")
            elif t == T_DESCEND - 1:
                report("after DESCEND", obj_start_xy)  # 拇指 xy 应 > 0.04（在方块外侧）
            elif t == T_INSERT - 1:
                report("after INSERT", obj_start_xy)   # pinch->obj 应 < 0.02 m
            elif t == T_CLOSE - 1:
                report("after CLOSE", obj_start_xy)
            elif t == T_LIFT - 1:
                report("after LIFT", obj_start_xy)     # obj_z 应约为 初始 + HOVER_HEIGHT
            elif t == T_TRAV2 - 1:
                place_report("after CARRY")            # 物体仍被举着，正移到目标上方
            elif t == T_LOWER - 1:
                place_report("after LOWER")            # 物体在桌面上方约 PLACE_DROP_CLEARANCE 处
            elif t == T_RELEASE - 1:
                place_report("after RELEASE")          # 物体应回到 rest_z、在目标上
            elif t == T_SIDE - 1 and q_pside is not None:
                place_report("after SIDE")             # 手掌已横移；物体必须"没有"被带动
            elif t == HORIZON - 1:
                place_report("after RETREAT")          # 手已撤离，物体静置在目标上

        # 成功 = 完整闭环：物体被放"在"目标上（xy 在容差内）、回到桌面
        # 高度（没被举着、也没掉下桌）、且处于静止（轻放，不是被甩出去的）。
        final_pos = obj.data.root_pos_w
        final_speed = torch.norm(obj.data.root_lin_vel_w, dim=-1)   # 物体末速度
        d_xy = torch.norm(final_pos[:, :2] - place_target_w[:, :2], dim=-1)   # 到目标的水平距离
        # 防范"回合中途环境自动重置"（例如噪声把方块撞下桌 -> object_fell 事件自动 reset）
        # 这些环境的回合时钟（以及记录的相位观测）在轨迹中途重启，即使最终状态碰巧通过检查，数据也
        # 已经被污染。干净的环境自我们 reset以来应恰好走了 HORIZON 步。
        no_reset = base_env.episode_length_buf == HORIZON
        if (~no_reset).any():
            print(f"[expert]   excluded {(~no_reset).sum().item()} env(s) that auto-reset mid-rollout")
        success = (
            (d_xy < PLACE_XY_TOL)
            # place_root_dz 会平移立放任务的期望最终高度（电钻站立时
            # root 比平躺高约 6.5 cm；被留成"平躺"在 rest_z 的电钻会
            # 正确地过不了这道门）。
            & ((final_pos[:, 2] - rest_z - PLACE_ROOT_DZ).abs() < PLACE_Z_TOL)
            & (final_speed < PLACE_SPEED_TOL)
            & no_reset
        )  # [N] 布尔张量，逐环境成功标记
        if UPRIGHT_CHECK:
            # 物体的 upright_axis（局部系）必须指向上方（识别翻倒的螺栓/倒回平躺的电钻：仅凭root高度不一定能区分"倒下"和"站立"）。
            z_local = torch.zeros_like(final_pos)
            z_local[:, 0], z_local[:, 1], z_local[:, 2] = UPRIGHT_AXIS
            up_dot = quat_apply(obj.data.root_quat_w, z_local)[:, 2]   # 局部轴旋到世界系后的 z 分量
            success = success & (up_dot > 0.94)  # 偏离竖直约 20 度以内
            print(f"[expert]   upright check: up-dot(mean)={up_dot.mean().item():.3f} "
                  f"(want > 0.94), {(up_dot > 0.94).sum().item()}/{num_envs} upright")
        rate = success.float().mean().item()
        fq = robot.data.joint_pos[:, finger_ids].abs().mean().item()
        print(f"[expert] rollout {r + 1}/{args_cli.rollouts}: PLACED {success.sum().item()}/{num_envs} "
              f"= {rate:.2%} | obj->target xy(mean)={d_xy.mean().item():.3f} m "
              f"| right-finger |q|(mean)={fq:.3f} rad")
        # 逐关节：指令目标 vs 实际角度（env 0）。手指若离目标很远，
        # 要么是顶住了物体（好事，说明在握持），要么是在和自己较劲。
        actual = robot.data.joint_pos[0, finger_ids].tolist()
        target = (closed_vec[0]).tolist()
        for nm, tg, ac in zip(RIGHT_FINGER_JOINTS, target, actual):
            print(f"[expert]     {nm:26s} target={tg:+.2f}  actual={ac:+.2f}")
        # 手臂跟踪检查（env 0）：下发的路点 vs 实际关节角。
        arm_actual = robot.data.joint_pos[0, arm_ids].tolist()
        arm_target = q_phover[0].tolist()
        for nm, tg, ac in zip(RIGHT_ARM_JOINTS, arm_target, arm_actual):
            print(f"[expert]     {nm:26s} target={tg:+.2f}  actual={ac:+.2f}")

        # 本回合数据搬到 CPU 累积（GPU 显存换 CPU 内存）
        all_obs.append(roll_obs.cpu())
        all_act.append(roll_act.cpu())
        all_success.append(success.cpu())

    # --- 保存数据集（只收录"成功"环境的完整轨迹） ---
    if args_cli.save_dataset is not None:
        obs_cat = torch.cat(all_obs, dim=1)       # [HORIZON, 总环境数, obs 维]
        act_cat = torch.cat(all_act, dim=1)       # [HORIZON, 总环境数, 动作维]
        succ_cat = torch.cat(all_success, dim=0)  # [总环境数] 成功掩码
        # 按成功掩码筛选环境，再把 (时间, 环境) 两维摊平成样本维
        obs_ok = obs_cat[:, succ_cat, :].reshape(-1, obs_cat.shape[-1]).numpy()
        act_ok = act_cat[:, succ_cat, :].reshape(-1, act_cat.shape[-1]).numpy()

        path = args_cli.save_dataset
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            # 文件已存在则"追加"：读入旧数据、拼接后整体重写
            prev = np.load(path)
            obs_ok = np.concatenate([prev["obs"], obs_ok], axis=0)
            act_ok = np.concatenate([prev["actions"], act_ok], axis=0)
        np.savez(path, obs=obs_ok, actions=act_ok)   # 存为 .npz：键 "obs" 和 "actions"
        print(f"[expert] saved dataset: {obs_ok.shape[0]} (obs, action) pairs -> {path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
