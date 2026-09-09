# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
搬运演示宿主脚本(transport demo host)。

三个里程碑(M1/M2/M3)在同一个脚本里串联：
- M1:走到停靠位(dock)并站稳——航点状态机把速度指令喂给行走 PPO 策略；
- M2(--bc_checkpoint):停靠后把"上半身"移交给 BC(行为克隆)抓放策略；
- M3(--carry_to_b):在 HOLD2 结束处切断 BC 周期（此刻工件已举到行走携带
  高度），冻结抓握，走到 B 桌上该工件放置区对应的停靠位，然后在那里用
  同一个 BC 时钟接着跑"放置腿"(place leg)。

状态机总流程：
    出生(spawn) -> [TURN] 原地转向目标 -> [NAV] 行走接近 -> [ALIGN] 精细
    对位 + 对准停靠朝向 -> [STAND] 指令清零、站立保持 -> 输出精度报告 ->
    [DONE] 右臂下放到 BC 起始位；只有当"实测"的手臂关节确实到位
    （位置门限 + 速度门限，且保持 --bc_settle 秒；见 BC_ARRIVE_* 常量）
    BC 才接管 -> [BC] 策略执行抓取周期（默认路径；与训练起始状态一致）。

    --bc_immediate 则直接从停靠位交接：写入的目标从抬臂位出发、在
    --bc_blend 秒内线性混合到 BC 指令。仅作诊断用途、保留作参考：
    2026-07-27 的实验证明,从抬臂位出发时右臂观测严重偏离训练分布(OOD),
    网络会锁死在 UP/TRAV 悬停吸引子里、始终不下探（整整 725 步 wrist_z
    停在 0.93-0.97,手指却按时刻表在半空合拢）——但该实验同时证明了
    "桌子不是诱因"（全程零桌面接触，周期依然没有执行）以及 BC 时钟确实
    严格按训练时刻表推进。

"""

import argparse
import math
import random
import re
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Transport demo M1: walk to a dock pose.")
# 必填：行走 PPO 策略的 skrl checkpoint（obs 69 / act 19，50 Hz）。
parser.add_argument("--walk_checkpoint", type=str, required=True)
# 要走向哪个静态停靠位：A = A 桌拾取位；B0..B2 = B 桌三个放置位。
parser.add_argument("--dock", type=str, default="A", choices=["A", "B0", "B1", "B2"])
# 动态停靠：按该工件的"实时位置"计算骨盆目标，使工件最终落在训练/验证时
# 的机器人相对位姿上（0.33 m 前方 / --dock_right 右侧）。
parser.add_argument("--target", type=str, default=None, choices=["cube", "bolt", "nut", "drill"])
# 螺母专用：停靠后螺母中心相对骨盆的"绝对右偏"（米）。默认 0.18 
parser.add_argument("--nut_dock_right", type=float, default=0.18)
# 电钻专用：停靠后电钻中心的"绝对右偏"（米）。默认 0.05 = BC 数据集采集时的几何
parser.add_argument("--drill_dock_right", type=float, default=0.05)
# cube/bolt 通用：停靠后工件的右偏（米），也是 M1 评分基准
parser.add_argument("--dock_right", type=float, default=0.05)
# A/B 隔离：'flat' = 训练场景（平地），用于二分"场景相关"的失败。
parser.add_argument("--scene", type=str, default="transport", choices=["transport", "flat"])
# 去掉四个工件（其碰撞体会刷 PhysX 报错，用此开关隔离该因素）。
parser.add_argument("--no_objects", action="store_true")
# STAND 阶段保持零指令的秒数，到时后才评 M1 精度分。
parser.add_argument("--stand_time", type=float, default=5.0)
# 仿真时间的全局超时上限（秒），超时直接中止。
parser.add_argument("--max_time", type=float, default=90.0)
# 每 N 个控制帧渲染一帧：物理与控制完全不受影响，调试 M4 长回路的后段任务时快进用。
parser.add_argument("--render_skip", type=int, default=1)
# 录制两个固定调试相机到带时间戳的视频：一个在 A 桌西侧中腿上方 1 m 朝东俯视 45°，一个在 B 桌南侧中腿上方
# 1 m 朝北俯视 45°、略向东偏转（把螺母的侧边放置收进画面）。隐含开启渲染
# 器的 --enable_cameras。文件落在 --video_dir，命名 <时间戳>_tableA/_tableB。
parser.add_argument("--record", action="store_true")
# （配合 --record）相机视频的输出目录，不存在时自动创建。
parser.add_argument("--video_dir", type=str, default="/opt/NVIDIA/IsaacLab/videos")
# （配合 --loop_all）只跑 7 任务表中的一部分：--jobs nut或 --jobs 6 / --jobs 4,7。被跳过的工件留
# 在 A 桌不动；放置点保持完整布局的分配
parser.add_argument("--jobs", type=str, default=None)
# （配合 --loop_all）批量统计：把（经 --jobs 过滤后的）任务表连跑 N 轮。
# 一轮在其最后一个任务得出判定（PLACED / NOT PLACED /DROPPED / TIMEOUT）的瞬间结束——跳过走回家，
# 环境重置、工件重生、下一轮从头开始。结束时打印逐任务成功率。
parser.add_argument("--repeat", type=int, default=1)
# （配合 --repeat）每轮里"每个任务"的仿真秒预算。卡死的一轮会在
# 任务数 x iter_cap 处被切断，未完成任务记 TIMEOUT——避免一次死锁
parser.add_argument("--iter_cap", type=float, default=180.0)
# （配合 --repeat）逐轮判定 CSV，每轮追加写入并立即关闭文件——抗崩溃，
parser.add_argument("--stats_csv", type=str, default=None)
# 行走期右肘增量（rad，叠加在默认 +0.87 上）；负值把前臂向上折起。
parser.add_argument("--raise_elbow", type=float, default=-1.60)
# 可选的右腕俯仰增量（rad），叠加到抬臂行走位上：当手掌够高但手指仍垂得
# 低时，用它给"指尖"额外的离桌间隙。
parser.add_argument("--raise_wrist_pitch", type=float, default=0.0)
# M2 入口：bc_train.py 训出的 bc_policy.pt（须与 --target 同一工件）。给出
# 后，停靠完成（DONE + 手臂到达 BC 起始位 + 稳定保持）就把上半身交给 BC
# 策略原地跑抓放周期，效果等同 bc_play，只是脚下由行走策略维持浮动底座平衡。
parser.add_argument("--bc_checkpoint", type=str, default=None)
# BC 时钟步数（100 Hz 语义），跑完后冻结最后的目标。830 = 抬臂起始的专家
# 周期长度（含加倍后的 FLIP 段）。M3 的切割在 LIFT 结束处、冻结抓握走去 B 桌。
parser.add_argument("--bc_steps", type=int, default=830)
# （仅分级交接路径）右臂必须保持"已验证到位"状态的秒数：7 个关节全部位于
# BC 起始位 BC_ARRIVE_POS_TOL 之内且速度低于 BC_ARRIVE_VEL_TOL。
parser.add_argument("--bc_settle", type=float, default=1.0)
# 诊断开关：跳过 DONE 的脚本化下放——BC 从停靠位直接接管手臂，目标从抬臂
# 位经 --bc_blend 秒混合过去。
parser.add_argument("--bc_immediate", action="store_true")
# 混合时长（秒）：写入的关节目标为(1-f)*交接瞬间姿态 + f*BC目标，f 从 0 线性升到 1。没有它，kp 300/400 的
# PD 会把抬起的手臂（肘 -0.73）一步拽到 BC 指令位。
parser.add_argument("--bc_blend", type=float, default=1.5)
# 配合 --bc_checkpoint：让网络在"抬臂行走位"接管手臂，走采集专家同款的
# GAIN 路径（跳过 DONE 低位下放）。
parser.add_argument("--bc_raised", action="store_true")
# 诊断：把喂给 BC 网络的前 200 个 tick 的精确 110 维观测 + 28 维输出存到 .pt。与 bc_play 的 --dump_obs
# 文件一起交给 bc_obs_diff.py：
parser.add_argument("--dump_obs", type=str, default=None)
# 让 BC 时钟从训练的第 N 个 tick 起跑而不是 0。
parser.add_argument("--bc_start_tick", type=int, default=0)
# bc_play --dump_obs 生成的参考文件。
parser.add_argument("--bc_ref_dump", type=str, default=None)
# 用"脚本化抓取专家"（grasp-expert.py 的先规划后执行，移植到站立底座）
parser.add_argument("--bc_expert", action="store_true")
# DAgger 采集：由网络（--bc_checkpoint）开车驱动手臂，脚本
# 化专家对每个到访状态回答作为标签。记录下(obs, expert_action)
parser.add_argument("--bc_dagger", action="store_true")
# M3：完整搬运。在A桌跑网络
parser.add_argument("--carry_to_b", action="store_true")
# B 桌放置区序号（0=蓝 x=5.85，1=绿 x=6.40，2=黄 x=6.95——即 PLACE_ZONE_XS 的顺序，+x 是布局草图里的最右侧）。
parser.add_argument("--place_zone", type=int, default=None, choices=[0, 1, 2])
# M4：全自动循环——一次运行把 A 桌全部七个工件搬到 B 桌，
# 按行序从 +y 端开始：cube, bolt, cube, drill, bolt, nut, drill。
parser.add_argument("--loop_all", action="store_true")
# （--loop_all 必填四件套）各工件类别的 BC checkpoint，逐任务切换加载。
parser.add_argument("--bc_cube", type=str, default=None,
                    help="(--loop_all) bc_policy.pt for the cube jobs.")
parser.add_argument("--bc_bolt", type=str, default=None,
                    help="(--loop_all) bc_policy.pt for the bolt jobs.")
parser.add_argument("--bc_nut", type=str, default=None,
                    help="(--loop_all) bc_policy.pt for the nut job.")
parser.add_argument("--bc_drill", type=str, default=None,
                    help="(--loop_all) bc_policy.pt for the drill jobs.")
# 配合 --bc_expert：若周期以"真实抬起 + PLACED"通过判定，把本次运行的
# (110 维 obs, 28 维 action) 行追加进该 .npz（bc_train.py 格式）。
parser.add_argument("--save_dataset", type=str, default=None)
# 配合 --bc_expert：每次查询给"执行"的原始 28 维动作加 N(0, noise)机器人
# 因此到访轻微偏离的状态，数据集教会网络往专家轨迹修正。采集配方：一遍干净 + 0.03 与 0.05 各一遍。
parser.add_argument("--action_noise", type=float, default=0.0)
# 采集模式：只生成 --target 工件、丢弃其余三个。
parser.add_argument("--only_target", action="store_true")
# 重置后立刻给目标工件出生 xy 加 ±J m 抖动
parser.add_argument("--spawn_jitter", type=float, default=0.0)
# 快速动作调参模式：重置后立刻把机器人传送到计算出的停靠位并从 STAND 开始跳过 TURN/NAV/ALIGN）
parser.add_argument("--spawn_at_dock", action="store_true")
# 实验性：BC 期间把 19 个行走关节 PD 冻结在交接时的站姿，而不是让行走策略持续配平。
parser.add_argument("--bc_statue", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# parse_known_args 会把不认识的开关静默吞进 hydra_args
_unknown_flags = [a for a in hydra_args if a.startswith("--")]
if _unknown_flags:
    parser.error(f"unrecognized arguments: {' '.join(_unknown_flags)} "
                 f"(running a stale copy of this script? hydra overrides are "
                 f"key=value, so '--' flags never belong in the remainder)")

# 把 hydra 覆盖参数还给 sys.argv，供下游框架解析。
sys.argv = [sys.argv[0]] + hydra_args
if args_cli.record:
    # RTX 传感器相机只有在应用启动"之前"告知渲染器才会被创建。
    args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 项目内导入（PROJECT_ROOT 必须在导入 tasks 之前设置好）----
import os
import time

PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.sensors import CameraCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_conjugate,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from tasks.common_config import G1RobotPresets
# 训练任务的观测/动作配置，原封不动复用（见模块 docstring 的 v2 架构说明）。
from tasks.rl.g1_locomotion.locomotion_env_cfg import (
    ACTION_JOINT_NAMES as WALK_ACTION_JOINT_NAMES,
    ActionsCfg,
    LocoSceneCfg,
    ObservationsCfg,
)
from tasks.rl.g1_transport.transport_scene_cfg import (
    BC_SPAWN_FWD,
    BC_SPAWN_RIGHT,
    DOCK_A_POS,
    DOCK_A_ROT,
    DOCK_B_POSES,
    DOCK_B_ROT,
    PICK_DOCK_AHEAD,
    PICK_DOCK_RIGHT,
    PLACE_SPOT_DX,
    PLACE_ZONE_XS,
    PLACE_ZONE_Y,
    ROBOT_START_POS,
    ROBOT_START_ROT,
    SURFACE_LIFT,
    TABLE_B_HALF_DEPTH,
    TABLE_B_HALF_LEN,
    TABLE_B_POS,
    TABLETOP_Z,
    TransportSceneCfg,
)

# 工件类别 -> 场景实体名的映射（cube 早于多工件场景出现，沿用了 pickplace环境的通用槽位名 "object"）
WORKPIECE_ENTITY = {"cube": "object", "bolt": "bolt", "nut": "nut", "drill": "drill"}
# 7 工件 M4 场景里的全部工件实体（--no_objects / --only_target 会遍历它）
ALL_WORKPIECE_ENTITIES = ("object", "bolt", "nut", "drill", "object2", "bolt2", "drill2")

CTRL_DT = 0.02                      # 环境控制周期（仿真 0.005 s x 抽取 4 = 50 Hz）

# ---- --record：固定调试相机 + 带时间戳的视频输出 ---------------------------
# 采样参数：30 fps、1 倍实时
REC_FPS = 30                        # 输出视频帧率（严格 30 fps）
REC_RES = (640, 480)                # 分辨率 (宽, 高)
CAM_ABOVE_TOP = 1.0                 # 相机高出桌面的高度（米，"沿桌腿向上 1 m"）
CAM_PITCH_DOWN_DEG = 45.0           # 光轴低于水平面 45 度
# B 桌相机从正北方向向东偏这么多度，好让螺母的东侧放置点（从南侧中腿看方位角约 24 度）留在画面内。
CAM_B_EAST_BIAS_DEG = 25.0


def _cam_quat(yaw_deg: float, pitch_down_deg: float):
    #世界约定（x 前 / z 上）下的相机四元数 (w,x,y,z)：先绕 +z 偏航，再把光轴向下俯仰 pitch_down_deg 度。
    hy = math.radians(yaw_deg) / 2.0
    hp = math.radians(pitch_down_deg) / 2.0
    cy, sy, cp, sp = math.cos(hy), math.sin(hy), math.cos(hp), math.sin(hp)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


class _VideoSink:
    #把 RGB 帧流式写到磁盘。后端在收到第一帧时按可用性选取：
    #imageio(+ffmpeg) mp4 -> OpenCV mp4 -> PNG 帧文件夹（总是可用的兜底）

    def __init__(self, base_path: str, fps: int):
        self.base = base_path       # 基础路径；扩展名由所选后端补上
        self.fps = fps
        self.backend = None
        self.writer = None
        self.path = None
        self.count = 0

    def _open(self, frame):
        #按 imageio -> cv2 -> PNG 的优先级挑选并打开写入后端。
        h, w = frame.shape[:2]
        try:
            import imageio.v2 as imageio
            self.path = self.base + ".mp4"
            self.writer = imageio.get_writer(self.path, fps=self.fps)
            self.backend = "imageio"
            return
        except Exception:
            pass
        try:
            import cv2
            self.path = self.base + ".mp4"
            self.writer = cv2.VideoWriter(
                self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.writer.isOpened():
                raise RuntimeError("cv2.VideoWriter would not open")
            self.backend = "cv2"
            return
        except Exception:
            pass
        # 该 python 环境没有任何视频编码器：退化为按序号导出 PNG
        self.path = self.base
        os.makedirs(self.path, exist_ok=True)
        self.backend = "png"

    def write(self, frame):
        #写一帧；首帧时惰性打开后端。
        if self.backend is None:
            self._open(frame)
        if self.backend == "imageio":
            self.writer.append_data(frame)
        elif self.backend == "cv2":
            import cv2
            self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        else:
            from PIL import Image
            Image.fromarray(frame).save(
                os.path.join(self.path, f"{self.count:06d}.png"))
        self.count += 1

    def close(self):
        #关闭写入器（PNG 后端无需收尾）。
        if self.backend == "imageio":
            self.writer.close()
        elif self.backend == "cv2":
            self.writer.release()


def _cam_frame(cam) -> np.ndarray:
    #取相机传感器最新的 RGB 帧，返回 HxWx3 uint8 数组。
    arr = cam.data.output["rgb"][0].detach().cpu().numpy()
    if arr.dtype != np.uint8:                     # 某些渲染管线输出 float
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return arr[..., :3]                           # 若带 alpha 通道则丢弃
# 速度指令钳位 = 训练时的采样范围；绝不下发策略从未见过的指令。
CMD_LIM_VX = (-0.4, 0.7)            # 前进上限压到训练最大 1.0 以下，接近时更平稳
CMD_LIM_VY = (-0.25, 0.25)          # 横向速度 = 训练采样范围
CMD_LIM_WZ = (-1.0, 1.0)            # 偏航角速度 = 训练采样范围

# ---- 停靠各阶段的门限 ----
NAV_DONE_DIST = 0.30                # 进入该半径即 NAV -> ALIGN
ALIGN_DONE_DIST = 0.04              # ALIGN 完成条件之一：位置误差 4 cm 以内...

ALIGN_DONE_YAW = 0.06               # ...且朝向误差在约 3.4 度以内
TURN_FACE_TOL = 0.35                # 大致朝向目标（rad）即 TURN -> NAV
PASS_DIST = 0.05                    # STAND 末尾的位置门限（无工件的停靠：dock B / flat）
PASS_YAW = 0.10                     # STAND 末尾的偏航门限（rad，约 5.7 度）

DOCK_AHEAD_TRIM = 0.03      # 瞄得更深：落点 fwd ~0.33（之前落在 0.296-0.31）
DOCK_RIGHT_TRIM = 0.02      # 瞄得更右：落点 lat ~-0.065（之前落在 -0.047）

PASS_AXIS = 0.035           # 逐轴工件位姿门限（BC 训练时的抖动是 ±3 cm）

PASS_AXIS_FWD_PICK = {"drill": 0.025, "nut": 0.025, "bolt": 0.025}

PASS_WINDOW_PICK = {
    "bolt": {"fwd": (-0.025, 0.010), "lat": (-0.035, 0.015),
             "yaw": (math.radians(-2.0), math.radians(0.3))},
}
# ---- 逐类别"仅瞄准"的前向修正----

DOCK_AHEAD_TRIM_PICK = {"nut": 0.0, "bolt": 0.005}

DOCK_AHEAD_TRIM_PLACE = {"drill": 0.05}

DOCK_RIGHT_TRIM_PLACE = {"drill": 0.02}
# ……并把电钻的 B 桌停靠前向门限收紧到 ±2.5 cm：仍然偏浅的落点会触发重停靠，而不是把电钻释放到棱角上。
PASS_AXIS_FWD_PLACE = {"drill": 0.025}

FLAT_GOAL_XY = (1.5, 0.5)
FLAT_GOAL_YAW = math.pi / 2

# ---- 右臂保持策略：行走时抬起，停靠后降到 BC 起始位 ----
# 行走策略"从设计上"既不驱动也不观测右臂，所以它的 PD 目标全程由本状态机掌管。

RAISE_RAMP_TIME = 1.5   # 秒：默认臂位 -> 抬臂行走位的 ramp 时长（PD 无冲击）
BC_RAMP_TIME = 2.0      # 秒：DONE 后抬臂位 -> BC 起始位的 ramp 时长（缓慢下放）
PALM_LINK = "right_hand_palm_link"      # 手臂遥测所用连杆（桌面 z = 0.794）

ALIGN_CMD_FLOOR = 0.12

ALIGN_CMD_FLOOR_Y = 0.18

ALIGN_CMD_FLOOR_WZ = 0.15

ALIGN_WZ_ESCALATE = 0.05        # 每卡住 2 秒地板加这么多
ALIGN_CMD_FLOOR_WZ_MAX = 0.35   # 偏航地板升级的上限

ALIGN_FLOOR_ENGAGE = 0.025

ALIGN_STUCK_SEC = 5.0           # 偏航冻结这么久 => 触发上述处理
ALIGN_STUCK_EPS = 0.005         # rad（约 0.3 度）；变化小于它即视为"冻结"

ALIGN_Y_ESCALATE = 0.03         # ALIGN 内每 2 秒 vy 地板加这么多
ALIGN_X_ESCALATE = 0.03         # ALIGN 内每 2 秒 vx 地板加这么多
ALIGN_CMD_FLOOR_X_MAX = 0.25    # 前向地板升级上限（贴近桌子时的安全）
ALIGN_MAX_SEC = 15.0            # 到时仍未收敛 => RETRY
RETRY_BACKOFF_DIST = 0.35       # 距停靠点超过该距离即结束 RETRY 后退
RETRY_MAX_SEC = 4.0             # 每次后退的安全时间上限（秒）
RETRY_VX = -0.30                # 后退速度（在 CMD_LIM_VX 的 -0.4 之内）

DOCK_RETRY_MAX = 5

GAIN_RAMP_TIME = 1.0

BC_ARRIVE_POS_TOL = 0.03    # rad：7 个右臂关节 |q - q_bc_start| 的最大允差
BC_ARRIVE_VEL_TOL = 0.10    # rad/s：右臂关节速度上限（高于它 = 仍在运动）
BC_ARRIVE_TIMEOUT = 8.0     # 秒：下放 ramp 结束后若手臂仍未到位（如被挡），
                            # 与其挂死不如带着逐关节残差的大声打印强行交接

# ============================================================
# M2：BC 策略交接接口。此处每个常量都"镜像"自
# tasks/rl/g1_pickplace（ActionsCfg / mdp/observations.py）
# 那个环境若有改动，这里必须同步改。
# ============================================================
# BC 动作项的 28 个关节。find_joints() 按"资产顺序"解析
# （preserve_order=False），与数据采集时动作项的解析方式完全一致，因此第k维动作在这里映射到同一个关节。
BC_ACTION_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint", "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]
BC_ACTION_SCALE = 0.5           # JointPositionActionCfg(scale=0.5, use_default_offset=True)；
                                # BC 环境里所有被写关节的默认位都是 0（base_fix 全零；
                                # 其非零的左臂保持位在本脚本里归行走策略管）
BC_MAX_EPISODE_STEPS = 800      # 回合长度 8.0 s / (dt 0.005 x decimation 2) = 800 步
# BC 环境的控制频率是 100 Hz，本宿主是 50 Hz：每个环境步把 BC 时钟推进
# 2 个 tick，轨迹就按"训练时的物理速度"播放（关节速度与训练一致；策略只是以一半频率采样观测）。
BC_CLOCK_PER_STEP = 2
BC_YAW = -math.pi / 2           # pickplace 机器人的朝向（init_rot Rz(-90)）。BC 观测的
                                # 相对向量是"世界系"的，所以这里的向量必须从本宿主的世界
                                # 系旋转到 BC 训练世界系。
BC_HAND_LINKS = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

# 110 维布局：joint_pos 0:29 | joint_vel 29:58 | hand_joints 58:72 |
# left_obj_rel 72:75 | right_obj_rel 75:78 | last_action 78:106 |
# phase 106 | place_target_rel 107:110。
BC_BODY_OBS_IDX = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18,
                   2, 5, 8, 11, 15, 19, 21, 23, 25, 27,
                   12, 16, 20, 22, 24, 26, 28]
BC_HAND_OBS_IDX = [31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38]
# BC 环境冻结在零位的关节（base_fix 的腿 + 腰）。本脚本里行走策略把它们保
# 持在站姿位——把这些"真实值"喂给 BC 网络会超出训练分布所以观测拼装时把它们清零（向训练分布做输入钳位）。
BC_FROZEN_JOINT_PATTERNS = [".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_.*"]
# 左臂观测钳位
BC_LEFT_ARM_OBS_POSE = {
    "left_shoulder_pitch_joint": 0.35,
    "left_shoulder_roll_joint": 0.18,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.87,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}
# 逐工件的放置目标采样盒 (right_lo, right_hi, fwd_lo, fwd_hi)，以工件为中
# 心、机器人系（下方采样器强制至少 3 cm 位移，与 bc_play 相同）。

BC_PLACE_BOXES = {
    # cube 右侧 0.04-0.10 -> 0.03-0.06：为解决"肘顶躯干"停靠右移了 4 cm，工件现在位于骨盆右侧约 0.09，旧的 +0.077
    # 采样会把放下点推到右侧 0.167超出舒适工作区。3-6cm 把放下点保持在右侧0.12-0.15，在bolt验证过的0.15预算内（距离可缩小）。
    "cube": (0.03, 0.06, 0.00, 0.03),
    "bolt": (0.04, 0.10, 0.00, 0.03),
    # nut 2026-07-29 用户规格：像 cube/bolt/drill 一样在同一深度向右放一
    # "小步"（右侧才是手臂够得着的方向）。旧的"向左/朝机器人"盒
    # 来自 grasp-expert 的右深出生角落，那次运行把放置目标放在左前方：phover 求解时 wrist_roll 又被钉在 -1.9
    #（残差 1.7 cm / 7.4 度）。
    "nut": (0.04, 0.08, 0.00, 0.03),
    # drill 2026-07-30（第 3 次运行，用户规格）：放下点仍是出生位右侧一
    # "小步"——这个偏移只是为了让释放的工件不在拾取原位，不是为了远行——
    # 从 4-6 cm 修到 3-5 cm。停靠回到 BC 名义 0.05 右侧后，合计正是
    # grasp-expert 验证过的数字：放下点在骨盆右侧 0.08-0.10，加上 8 cm 撤
    # 离侧移最大 0.18 = 在第 41 轮的横向预算内。（第 2 次运行的"出生位左
    # 侧"盒是停靠 0.15 时预算溢出的权宜之计，随停靠回退一起退役。）
    "drill": (0.03, 0.05, 0.00, 0.03),
}
# 专家相位边界（BC 时钟步），用于与 bc_play 的打印 1:1 对照的遥测。
BC_PHASE_MARKS = {19: "HOLD", 104: "TRAV", 194: "FLIP", 274: "DESCEND",
                  334: "INSERT", 424: "CLOSE", 504: "LIFT", 609: "CARRY",
                  689: "LOWER", 739: "RELEASE", 764: "SIDE", 829: "RETREAT"}
# 成功判据，镜像自 bc_play（"直立轴"检查留给 GUI 肉眼确认）。
BC_PLACE_XY_TOL = 0.04          # 工件到目标的水平距离门限（m）
BC_PLACE_Z_TOL = 0.03           # 工件相对静置高度的 z 偏差门限（m）
BC_PLACE_SPEED_TOL = 0.10       # 评分时工件速度门限（m/s，仍在动 = 不算放好）
# 判定"真抓起"所需的最小峰值抬升（相对静置高度）——用来区分真正的
# GRASP+LIFT 和把工件沿桌面推过去（第 10 次运行）。专家 hover/carry 在
# 0.10 时此值是 0.05；
BC_REAL_LIFT_MIN = 0.03
# 逐工件的"放置后根部高度差"：drill 立式放置后根部比躺姿高 0.065 m。
BC_PLACE_ROOT_DZ = {"cube": 0.0, "bolt": 0.0, "nut": 0.0, "drill": 0.065}
# 行走动作项的 scale（locomotion_env_cfg 的 ActionsCfg）。用于合成"雕像模
# 式"里复现捕获站姿的常量动作：JointPositionAction 目标 = 默认位 + scalex 动作。
WALK_ACTION_SCALE = 0.25

# ---- 站立底座上的脚本化专家（--bc_expert）----------------------------------
# grasp-expert.py 的"先规划后执行"周期
EXPERT_PROFILES = {
    # "窗口居中"的教训
    "cube": dict(grasp_right=0.02, grasp_fwd=0.0, grasp_up=0.005, hover=0.06,
                 obj_half=0.03, thumb_margin=0.015, close_fraction=0.46,
                 arm_range=(0.4, 0.25, 0.2, 6.28, 0.25, 1.2, 0.4),
                 ik_ori_weight=0.15, drop_clearance=0.018),
    # bolt：基于 cube 教训的预调参（当时未实测，首个 bolt 运行待跑）：
    # hover 0.09 -> 0.06（bolt 在杆身向上 10.5 cm 处抓取，高悬停会更接近抬
    # 臂起始高度，像 cube 一样让手掌上弓）；drop_clearance 保持很低
    # （0.008）——此处手腕位于桌面上方 10.5 cm，不可能刮擦，而竖立的螺栓
    # 若在 1.8 cm 高处释放会翻倒。
    "bolt": dict(grasp_right=0.010, grasp_fwd=0.0, grasp_up=0.105, hover=0.06,
                 obj_half=0.025, thumb_margin=0.01, close_fraction=0.50,
                 arm_range=(0.4, 0.25, 0.2, 6.28, 0.25, 1.2, 0.4),
                 ik_ori_weight=0.15, drop_clearance=0.008),
    # nut = grasp-expert
    
    "nut": dict(approach="top", grasp_right=0.0, grasp_fwd=-0.09, grasp_up=0.11,
                hover=0.06, obj_half=0.1125, thumb_margin=0.01, insert_clear=0.06,
                close_fraction=0.33, preclose=0.5,
                arm_range=(1.0, 0.25, 0.25, 6.28, 1.9, 1.0, 1.2),
                ik_ori_weight=0.6, drop_clearance=0.005),
    # drill = grasp-expert 
    "drill": dict(approach="top", grasp_pitch=0.262,
                  grasp_right=0.0, grasp_fwd=0.0, grasp_up=0.02, hover=0.05,
                  obj_half=0.03, thumb_margin=0.01, insert_clear=0.08,
                  close_fraction=0.45, preclose=0.0,
                  arm_range=(1.3, 0.25, 0.25, 6.28, 1.9, 1.0, 1.55),
                  ik_ori_weight=0.6, drop_clearance=0.005,
                  place_reorient=True, place_root_dz=0.065, place_pitch=0.14,
                  retreat_side=0.08, place_shove_comp=0.02),
}

CARRY_PINCH_Z = 0.95

EXPERT_T = dict(HOLD=20, UP=20, TRAV=105, FLIP=195, DESCEND=275, INSERT=335,
                SETTLE=355, CLOSE=425, LIFT=505, HOLD2=550, TRAV2=610,
                LOWER=690, RELEASE=740, SIDE=765, RETREAT=800, END=830)
# DLS IK 求解器参数：最多 400 次迭代、阻尼 0.05、单步限幅 0.05 rad、位置
# 容差 5 mm、角度容差 5 度；null_gain 是零空间向 q_zero 牵引的逐关节增益。
EXPERT_IK = dict(iters=400, damping=0.05, step=0.05, pos_tol=0.005,
                 ang_tol_deg=5.0, null_gain=(0.3, 0.5, 0.5, 0.02, 0.02, 0.02, 0.02))
# 周期后的返回，四种工件通用：END 之后手臂先在撤离
# 顶点短暂沉降，再 ramp 回抬臂行走位——这样最终 M3 的后撤/转身/行走都从验
# 证过的携带姿态出发、零高度变化。该段"从构造上"不属于 BC 数据：记录在
# --bc_steps（== END）处停止，只有目标"写入"在额外的返回窗口里继续。
BC_RETURN_SETTLE = 20    # END 后在撤离顶点保持的 tick 数
BC_RETURN_TICKS = 100    # 撤离顶点 -> 行走抬臂位 ramp 的 tick 数（1 秒）

# ---- M3（--carry_to_b）：A 桌拾取 -> 搬运 -> B 桌放置 -----------------------
# 切割 tick，逐工件。默认 = HOLD2 结束（tick 550）：手臂保持行走携带姿态
# DRILL 改在 TRAV2/CARRY 结束处切割
M3_CUT_TICK = {"cube": EXPERT_T["HOLD2"], "bolt": EXPERT_T["HOLD2"],
               "nut": EXPERT_T["HOLD2"], "drill": EXPERT_T["TRAV2"]}
# 工件 -> 放置区预设（PLACE_ZONE_XS 的下标：0=蓝 x=5.85，1=绿 x=6.40，
# 2=黄 x=6.95）。2026-08-08 晚间用户规格：cube -> 蓝，bolt -> 绿（中间），
# drill -> 黄；nut 没有区域——它去 B 桌"东侧短边"的中点（见下方 M3_NUT_*）。
# --place_zone 可覆盖（并强制螺母也走区域停靠）。
M3_PLACE_ZONE = {"cube": 0, "bolt": 1, "drill": 2, "nut": 2}
# 螺母放置点（"正对黄色框的那条桌边的中点"）：位于东侧短边的中心线上，从
# B 桌 +x 端面向 -x 进近。距桌沿深度 0.25 m  = 红色
# 参考点（NUT_SIDE_REF_XY）：螺母正好落在点上。可达性：训练相对位姿把深度
# 上限限制在 ~0.385（骨盆最近可到离桌沿 ~0.13 m，螺母在前方 0.515）。
# 请在查看器中核实：假设 2.2 m 的桌子（东沿 x=7.5）。
M3_NUT_PLACE_XY = (TABLE_B_POS[0] + 1.10 - 0.25, TABLE_B_POS[1])
M3_NUT_DOCK_YAW = math.pi          # 面向 -x；机器人右侧 = +y（北）

M3_STAGE_DIST = 1.0
M3_WP_RADIUS = 0.30       # 中间航点的到达半径（米）
# 最后一个航点（整备点）的到达：更紧的半径 + 与距离成正比的减速。

M3_WP_RADIUS_LAST = 0.15
# TRANSIT 的"原地转"阈值：朝向误差超过它就停下原地旋转、而不是画弧
# （vx 0.5 / wz 0.6 = 0.83 m 转弯半径——nut 运行曾为了画弧切进一个偏离朝向
# 90 度的航点，围着桌子东端绕了三圈）。
M3_TRANSIT_TURN_ERR = 1.0
# 搬运腿的步态上限：工件挂在一条行走策略"既不观测也不控制"的僵硬前伸手臂
# 上——限制指令包络，让携带质量只是温和扰动而不是甩鞭。（空手行走验证到
# 0.7 m/s；先保守起步。）
M3_CARRY_VX_CAP = 0.5      # 搬运时前进速度上限（m/s）
M3_CARRY_WZ_CAP = 0.6      # 搬运时偏航角速度上限（rad/s）
# 螺母搬运调整
M3_CARRY_WZ_CAP_NUT = 0.35
M3_CARRY_VY_CAP_NUT = 0.10
# 搬运握紧：手指是位置控制的，所以握力 =对工件表面的 PD 误差。

M3_CARRY_GRIP_KP_SCALE = 2.0
# 切割后的后撤：先笔直后退这么远再转向 B 桌，使约 0.35 m 前伸手臂 + 工件
# 的扫掠在约 180 度转身时避开 A 桌（工件在 z~0.92+，桌面 0.80——余量在竖直
# 方向也有，但后退让扫掠"平凡地"安全）。
M3_BACKOFF_DIST = 0.5      # 后撤距离（米）
M3_BACKOFF_MAX_SEC = 6.0   # 后撤的安全时间上限（秒）

# ---- M4（--loop_all）：一次运行搬完全部七个工件 ----------------------------
# 任务表：(场景实体, 类别)，搬运顺序 = A 桌行序、从 +y 端开始
# 放置点按任务推导：cube/bolt/drill 成对去各自类别区域的中央深度，第一件在中心以东
# PLACE_SPOT_DX、第二件以西同距离（先东后西——在西侧停靠时，已放好的东侧
# 工件位于机器人"左侧"，避开右臂放下时的扫掠）；唯一的 nut 保持其东侧位置。
M4_JOBS = (
    ("object", "cube"),
    ("bolt", "bolt"),
    ("object2", "cube"),
    ("drill", "drill"),
    ("bolt2", "bolt"),
    ("nut", "nut"),
    ("drill2", "drill"),
)

M4_A_RESTAGE_X = 0.37 + 2.5     # A 桌整备点的 x（进近边 + 2.5 m）
M4_B_EXIT_Y = TABLE_B_POS[1] + TABLE_B_HALF_DEPTH + 1.08          # 撤离走廊 y，约 0.45
M4_B_EXIT_X_EAST = TABLE_B_POS[0] + TABLE_B_HALF_LEN + 1.10       # 东撤离纵线 x，约 8.6
M4_B_EXIT_X_WEST = TABLE_B_POS[0] - TABLE_B_HALF_LEN - 0.50       # 西走廊出口 x，约 4.8
M4_ARM_RETURN_SEC = 1.5   # 放置末姿态 -> 行走抬臂位的 ramp 时长（smoothstep）
M4_REPORT_HOLD_TICKS = 100  # M2 报告后再保持这么多 BC tick 才开始移动

M4_DROP_GRIP = 0.55    # m：搬运期间 手->物体 距离超过它 = 掉了
M4_DROP_OBJ_Z = 0.50   # m：搬运期间物体高度低于它 = 掉了（桌面约 0.75）


def _ease(p: float) -> float:
    #smoothstep 缓动：每段 ramp 的两端速度为零（p^2 * (3 - 2p)）。
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


def m3_carry_route_for(spot_xy, dock_yaw: float, aim_ahead_b: float,
                       nut_side: bool) -> list:
    #为一个放置点生成航点路线（A 桌 -> B 桌停靠位）。

    # 停靠轴的前向单位向量；整备点 = 放置点沿该轴后退 (aim_ahead_b + 1 m)
    fbx, fby = math.cos(dock_yaw), math.sin(dock_yaw)
    staging = (spot_xy[0] - (aim_ahead_b + M3_STAGE_DIST) * fbx,
               spot_xy[1] - (aim_ahead_b + M3_STAGE_DIST) * fby)
    if nut_side:
        # 螺母走北侧走廊：先到桌子西端外的走廊入口，沿走廊到整备点正北，再拐到整备点
        corridor_y = TABLE_B_POS[1] + TABLE_B_HALF_DEPTH + 1.0
        return [(TABLE_B_POS[0] - TABLE_B_HALF_LEN - 0.30, corridor_y),
                (staging[0], corridor_y),
                staging]
    return [staging]


class ScriptedExpert:
    #grasp-expert.py 的抓放周期（cube/bolt侧向进近、nut掌心向下顶部进近），移植到"浮动"底座上。

    #与固定底座原版的刻意差异：
    #- 起始姿态 = 抬起的行走手臂（交接时实时捕获），而不是全零 BC 起始位/wholebody 默认位。
    #  第一段运动是 抬臂 -> 抓取点上方悬停（无低位下放、无 UP 抬升）
    #  用户规格：规避 M1 指尖碰工件的风险，也腾出工件上方的工作空间；
    #- 雅可比索引适配浮动底座（前面多6个底座自由度列，连杆下标不再偏移）；
    #- 航点在"世界系"里按实时底座/物体位姿规划；站立底座只有毫米级漂移，且
    #  TRAV 结束时的重规划会按新的物体位置重新瞄准下探，与原版完全一致；
    #- 输出是原始的 28 维 BC 动作（目标 = 0.5 x 动作），走 transport 现成的
    #  BC 写入通路，因此记录下的 (obs, action) 行"就是"部署分布本身。
    

    # 右臂 7 个关节（IK 的操作对象），顺序固定
    ARM_JOINTS = [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    FINGER_JOINTS = [  # 顺序与下面的 FINGER_CLOSED 一一对应（拇指、中指、食指）
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
    ]
    # 各手指关节"完全闭合"时的角度（rad）；实际闭合量 = 它乘 close_fraction
    FINGER_CLOSED = [0.0, -1.047, -1.745, 1.571, 1.745, 1.571, 1.745]
    # 指尖连杆：拇指尖、食指尖、中指尖（用于测捏点/拇指偏移）
    TIP_LINKS = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]

    def __init__(self, robot, target_obj, device, profile, act_names):
        #robot/target_obj 是场景实体；profile是EXPERT_PROFILES里的一项；act_names 是BC动作项按实际解析顺序排列的关节名。
        self.robot, self.obj, self.dev, self.p = robot, target_obj, device, profile
        # BC 动作槽位按"排序后的关节顺序"排列（JointActionCfg.preserve_order 默认 False），不是名义上的 ACTION_JOINT_NAMES 顺序。
       
        slot_of = {n: i for i, n in enumerate(act_names)}
        self.arm_slots = torch.tensor([slot_of[n] for n in self.ARM_JOINTS],
                                      device=device, dtype=torch.long)
        self.fin_slots = torch.tensor([slot_of[n] for n in self.FINGER_JOINTS],
                                      device=device, dtype=torch.long)
        self.arm_ids, _ = robot.find_joints(self.ARM_JOINTS, preserve_order=True)
        self.fin_ids, _ = robot.find_joints(self.FINGER_JOINTS, preserve_order=True)
        self.ee_id = robot.find_bodies(PALM_LINK)[0][0]
        self.tip_ids = [robot.find_bodies(n)[0][0] for n in self.TIP_LINKS]
        self.jac_link = self.ee_id - 1 if robot.is_fixed_base else self.ee_id
        col0 = 0 if robot.is_fixed_base else 6   # 浮动底座：雅可比前 6 列是底座自由度
        self.jac_cols = [col0 + j for j in self.arm_ids]
        # q_zero = 验证过的 BC 起始姿态（右臂全零）：IK 窗口、零空间牵引和掌向目标"全部"居中于此
        # q_start = 交接时实时捕获的行走抬臂位；"只"用作 TRAV ramp 的起点，绝不是 IK 约束。
        self.q_zero = torch.zeros((1, len(self.arm_ids)), device=device)
        self.q_start = torch.zeros((1, len(self.arm_ids)), device=device)
        limits = getattr(robot.data, "joint_pos_limits", None)
        if limits is None:
            limits = robot.data.joint_limits
        arm_limits = limits[:1, self.arm_ids]        # [1, 7, 2] 硬限位
        lo = arm_limits[..., 0].clone()
        hi = arm_limits[..., 1].clone()
        # IK 搜索窗 = 硬限位与"q_zero ± arm_range"的交集（逐关节收紧）
        for j, rng in enumerate(profile["arm_range"]):
            lo[:, j] = torch.maximum(lo[:, j], self.q_zero[:, j] - rng)
            hi[:, j] = torch.minimum(hi[:, j], self.q_zero[:, j] + rng)
        self.arm_lo, self.arm_hi = lo, hi
        self.null_gain = torch.tensor(EXPERT_IK["null_gain"], device=device).unsqueeze(0)
        # 目标闭合姿态 = 完全闭合角 x 该工件的 close_fraction
        self.closed = (torch.tensor(self.FINGER_CLOSED, device=device).unsqueeze(0)
                       * profile["close_fraction"])
        self.approach = profile.get("approach", "side")
        # 顶部进近穿孔用的"预闭合"姿态：横移/下探期间食指/中指部分卷曲，
        # 拇指完全张开（它必须从物体"外侧"经过；卷曲的拇指会压到物体顶上）。
        # 侧向进近的 preclose=0 -> 全零（无操作）。
        self.preclose = self.closed * profile.get("preclose", 0.0)
        self.preclose[:, :3] = 0.0   # [thumb0, thumb1, thumb2] 保持张开直到 CLOSE
        self.eye6 = torch.eye(6, device=device).unsqueeze(0)
        self.eye7 = torch.eye(len(self.arm_ids), device=device).unsqueeze(0)
        self.pinch_local = torch.zeros((1, 3), device=device)   # 捏点在掌系的偏移
        self.thumb_local = torch.zeros((1, 3), device=device)   # 拇指尖在掌系的偏移
        self.fwd = torch.zeros((1, 3), device=device)           # 机器人->物体的水平前向
        self.rest_quat = None            # 零位掌向四元数（侧向进近的姿态目标）
        self.flip_quat = None            # = rest（侧向）| 纯掌心向下翻转（顶部）
        self.grasp_quat = None           # = flip 再叠加可选的 grasp_pitch 前倾
        self.q_hover = self.q_flip = self.q_tilt = None
        self.q_desc = self.q_grasp = self.q_carry = None
        self.q_phover = self.q_pdown = self.q_pside = self.q_pside_up = None
        self.grasp_z = None
        self.obj_rest_z = 0.0            # 静置根部高度（立式放置的高度参考）
        self.planned_pick = self.replanned = self.planned_place = False

    def capture_start_pose(self):
        #把"实时"的抬臂手臂冻结为 TRAV ramp 的起点（交接时调用一次）。
        #刻意不动IK窗口或零空间参考——它们保持在q_zero上

        self.q_start = self.robot.data.joint_pos[:1, self.arm_ids].clone()
        elb = self.q_start[0, 3].item()
        print(f"[demo]   expert start = RAISED walk pose "
              f"(elbow {elb:+.3f}; first motion -> hover above grasp, no low descent; "
              f"IK posture reference stays at the zero BC-start pose)")

    # ---- 运动学辅助（"写入-恢复"FK 技巧，与 grasp-expert 相同）----
    def _fk_palm(self):
        """取当前手掌位姿 + 右臂 7 列雅可比（供 IK 迭代用）。"""
        jac_full = self.robot.root_physx_view.get_jacobians()
        tf = self.robot.root_physx_view.get_link_transforms()
        palm_pos = tf[:, self.ee_id, :3]
        palm_quat = tf[:, self.ee_id, 3:7][:, [3, 0, 1, 2]]  # physx xyzw -> wxyz
        # 两步索引（避免基础/高级索引混用的歧义）：形状 [1, 6, 7]
        jac = jac_full[:, self.jac_link][:, :, self.jac_cols]
        return palm_pos, palm_quat, jac

    def measure_offsets(self):
        #测量掌系下的两个偏移：张开的拇指尖 + "虚拟闭合"的捏取中心。
        #做法：把手指瞬间写成闭合姿态取FK，再原样恢复——物理不前进。
        robot = self.robot
        q_snap = robot.data.joint_pos.clone()
        qd_snap = robot.data.joint_vel.clone()
        zero = torch.zeros_like(qd_snap)
        robot.root_physx_view.get_jacobians()
        tf = robot.root_physx_view.get_link_transforms()
        palm_pos = tf[:, self.ee_id, :3]
        palm_quat = tf[:, self.ee_id, 3:7][:, [3, 0, 1, 2]]
        # 张开状态下拇指尖相对手掌的偏移（顶部进近避让计算要用）
        off_t, _ = subtract_frame_transforms(palm_pos, palm_quat, tf[:, self.tip_ids[0], :3])
        self.thumb_local[:] = off_t[:1]
        # 把手指虚拟写成闭合位，取捏取中心（拇指尖与食/中指尖中点的中点）
        q_work = q_snap.clone()
        q_work[:, self.fin_ids] = self.closed
        robot.write_joint_state_to_sim(q_work, zero)
        robot.root_physx_view.get_jacobians()   # 强制刷新运动学
        tf = robot.root_physx_view.get_link_transforms()
        thumb = tf[:, self.tip_ids[0], :3]
        fingers = 0.5 * (tf[:, self.tip_ids[1], :3] + tf[:, self.tip_ids[2], :3])
        pinch_centre = 0.5 * (thumb + fingers)
        palm_pos = tf[:, self.ee_id, :3]
        palm_quat = tf[:, self.ee_id, 3:7][:, [3, 0, 1, 2]]
        off, _ = subtract_frame_transforms(palm_pos, palm_quat, pinch_centre)
        self.pinch_local[:] = off[:1]
        robot.write_joint_state_to_sim(q_snap, qd_snap)
        robot.root_physx_view.get_jacobians()
        print(f"[demo]   expert pinch offset (palm frame): "
              f"{[round(v, 3) for v in off[0].tolist()]} (|.|={off[0].norm().item():.3f} m)")

    def pinch_w(self):
        #当前捏取中心的世界坐标（手掌位姿 + 掌系捏点偏移）。
        r = self.robot
        return (r.data.body_pos_w[:1, self.ee_id]
                + quat_apply(r.data.body_quat_w[:1, self.ee_id], self.pinch_local))

    def _right(self):
        #机器人右方向的水平单位向量（由 fwd 顺时针旋转 90 度得到）。
        f = self.fwd
        r = torch.stack([f[:, 1], -f[:, 0], torch.zeros_like(f[:, 0])], dim=-1)
        return torch.nn.functional.normalize(r, dim=-1)

    def _grasp_point(self):
        #抓取点 = 物体实时位置 + 档案里的 右/前/上 偏移。
        gp = (self.obj.data.root_pos_w[:1].clone()
              + self._right() * self.p["grasp_right"] + self.fwd * self.p["grasp_fwd"])
        gp[:, 2] += self.p["grasp_up"]
        return gp

    def _desc_point(self, gp):
        #下探点（DESCEND 的终点）。
        #侧向：在工件旁边、向机器人右侧平移，使"张开的拇指"避开其右侧面；之后 INSERT 向左滑入。
        #顶部：抓取点正上方 insert_clear 处（预闭合的手指从孔上方开始；INSERT 竖直下压）。
        if self.approach == "top":
            d = gp.clone()
            d[:, 2] += self.p["insert_clear"]
            return d
        # 侧向：算出拇指尖将落在哪里，若离工件右侧面不足
        # obj_half + thumb_margin 就向右补平移量
        palm_t = gp - quat_apply(self.grasp_quat, self.pinch_local)
        thumb_t = palm_t + quat_apply(self.grasp_quat, self.thumb_local)
        rel = ((thumb_t - gp) * self._right()).sum(dim=-1, keepdim=True)
        shift = torch.clamp((self.p["obj_half"] + self.p["thumb_margin"]) - rel, min=0.0)
        print(f"[demo]   expert PLAN shift: thumb side-shift={shift[0].item():.3f} m")
        return gp + self._right() * shift

    def _solve_ik(self, pinch_target_w, q_start, label, quat_w=None, ori_weight=None):
        #右臂的 DLS（阻尼最小二乘）IK：把"捏取中心"放到目标点，掌向被拉向quat_w
        #(默认 grasp_quat：侧向进近用零位静息掌向、权重 0.15；顶部进近用掌心向下的翻转、权重 0.6——那里"翻掌朝下"本身就是任务)
        #顶部进近翻转前的悬停会以"低权重"传 quat_w=rest_quat，让横移期间手掌保持朝左。
        #运动学写入-恢复：物理从不前进，屏幕上什么都不动。
        robot = self.robot
        q_snap = robot.data.joint_pos.clone()
        qd_snap = robot.data.joint_vel.clone()
        zero = torch.zeros_like(qd_snap)
        target_quat = self.grasp_quat if quat_w is None else quat_w
        # 捏点目标换算成手掌目标（减去掌系捏点偏移旋到世界系）
        palm_target_w = pinch_target_w - quat_apply(target_quat, self.pinch_local)
        ang_tol = math.radians(EXPERT_IK["ang_tol_deg"])
        ori_w = self.p["ik_ori_weight"] if ori_weight is None else ori_weight
        q_arm = q_start.clone()
        q_work = q_snap.clone()
        for _ in range(EXPERT_IK["iters"]):
            # 把候选臂角写进仿真取 FK（写入-恢复，物理不前进）
            q_work[:, self.arm_ids] = q_arm
            robot.write_joint_state_to_sim(q_work, zero)
            palm_pos, palm_quat, jac = self._fk_palm()
            pos_err = palm_target_w - palm_pos
            # 姿态误差 = 目标四元数 x 当前共轭，转轴角向量
            ang_err = axis_angle_from_quat(quat_mul(target_quat, quat_conjugate(palm_quat)))
            if pos_err.norm(dim=-1).max() < EXPERT_IK["pos_tol"] and ang_err.norm(dim=-1).max() < ang_tol:
                break
            # 6 维任务误差 = [位置; 权重 x 姿态]；DLS：dq = J^T (JJ^T + λ²I)^-1 e
            err = torch.cat([pos_err, ori_w * ang_err], dim=-1).unsqueeze(-1)
            jjt = jac @ jac.transpose(1, 2) + (EXPERT_IK["damping"] ** 2) * self.eye6
            dq = (jac.transpose(1, 2) @ torch.linalg.solve(jjt, err)).squeeze(-1)
            # 零空间项：在不影响任务误差的方向上把姿态往 q_zero 拉
            jpinv = jac.transpose(1, 2) @ torch.linalg.inv(jjt)
            null_proj = self.eye7 - jpinv @ jac
            dq = dq + (null_proj @ (self.null_gain * (self.q_zero - q_arm)).unsqueeze(-1)).squeeze(-1)
            # 步长限幅 + 关节窗口（q_zero ± arm_range 与硬限位的交集）
            q_arm = torch.clamp(q_arm + torch.clamp(dq, -EXPERT_IK["step"], EXPERT_IK["step"]),
                                self.arm_lo, self.arm_hi)
        q_work[:, self.arm_ids] = q_arm
        robot.write_joint_state_to_sim(q_work, zero)
        palm_pos, palm_quat, _ = self._fk_palm()
        pinch_pos = palm_pos + quat_apply(palm_quat, self.pinch_local)
        res_vec = pinch_target_w - pinch_pos
        res_a = torch.norm(axis_angle_from_quat(quat_mul(target_quat, quat_conjugate(palm_quat))), dim=-1)
        print(f"[demo]   expert PLAN {label:6s}: residual pos={res_vec[0].norm().item():.4f} m "
              f"xyz={[round(v, 3) for v in res_vec[0].tolist()]} | "
              f"ang={math.degrees(res_a[0].item()):.1f} deg | "
              f"q_arm={[round(v, 3) for v in q_arm[0].tolist()]}")
        if res_vec[0].norm().item() > 0.02 or math.degrees(res_a[0].item()) > 10.0:
            print(f"[demo]   WARNING: {label} IK off-target - INSERT/place will undershoot; "
                  f"widen arm_range or ease the aim")
        robot.write_joint_state_to_sim(q_snap, qd_snap)
        robot.root_physx_view.get_jacobians()
        return q_arm

    def _palm_quat_at_zero(self):
        #手臂"虚拟地"处于零位（BC 起始）姿态时的掌向——前臂水平，即验证过的侧向进近方向。写入-恢复 FK，物理不步进。
    
        robot = self.robot
        q_snap = robot.data.joint_pos.clone()
        qd_snap = robot.data.joint_vel.clone()
        q_work = q_snap.clone()
        q_work[:, self.arm_ids] = self.q_zero
        robot.write_joint_state_to_sim(q_work, torch.zeros_like(qd_snap))
        robot.root_physx_view.get_jacobians()
        tf = robot.root_physx_view.get_link_transforms()
        quat = tf[:1, self.ee_id, 3:7][:, [3, 0, 1, 2]].clone()
        robot.write_joint_state_to_sim(q_snap, qd_snap)
        robot.root_physx_view.get_jacobians()
        return quat

    # ---- 两条腿的规划（拾取腿 / 放置腿）----
    def plan_pick(self):
        #规划拾取腿的全部 IK 航点（hover/flip/tilt/desc/grasp/carry）。
        self.measure_offsets()
        # fwd = 机器人指向物体的水平单位向量（进近方向）
        to_obj = self.obj.data.root_pos_w[:1] - self.robot.data.root_pos_w[:1]
        to_obj[:, 2] = 0.0
        self.fwd[:] = torch.nn.functional.normalize(to_obj, dim=-1)
        # 掌向目标：从"零位"的手掌方向出发。
        # 侧向：直接沿用。顶部：绕机器人前向 roll -90 度——掌心法线从机器人左侧转为竖直向下，手指仍指向前方
        self.rest_quat = self._palm_quat_at_zero()
        if self.approach == "top":
            roll = torch.full((1,), -math.pi / 2, device=self.dev)
            self.flip_quat = quat_mul(quat_from_angle_axis(roll, self.fwd), self.rest_quat)
            # drill：在翻转之上再绕机器人右向加 15 度前倾（grasp_pitch）——
            # 只有 desc/grasp 瞄它；FLIP 步本身目标是纯掌心向下姿态，之后
            # 一小段 TILT 子 ramp 才加上前倾（grasp-expert 第 33 轮）。
            pitch = self.p.get("grasp_pitch", 0.0)
            if pitch:
                ang = torch.full((1,), -pitch, device=self.dev)
                self.grasp_quat = quat_mul(quat_from_angle_axis(ang, self._right()), self.flip_quat)
            else:
                self.grasp_quat = self.flip_quat
        else:
            self.flip_quat = self.grasp_quat = self.rest_quat
        self.obj_rest_z = self.obj.data.root_pos_w[0, 2].item()
        gp = self._grasp_point()
        self.grasp_z = gp[:, 2].clone()
        desc = self._desc_point(gp)
        # 抬臂起始：没有UP航点——TRAV直接 ramp q_start -> q_hover。
        
        hov = desc.clone()
        hov[:, 2] += self.p["hover"]
        if self.approach == "top":
            self.q_hover = self._solve_ik(hov, self.q_zero, "hover",
                                          quat_w=self.rest_quat, ori_weight=0.15)
            self.q_flip = self._solve_ik(hov, self.q_hover, "flip", quat_w=self.flip_quat)
            if self.p.get("grasp_pitch", 0.0):
                self.q_tilt = self._solve_ik(hov, self.q_flip, "tilt")
        else:
            self.q_hover = self._solve_ik(hov, self.q_zero, "hover")
            self.q_flip = self.q_hover
        pre_desc = self.q_tilt if self.q_tilt is not None else self.q_flip
        self.q_desc = self._solve_ik(desc, pre_desc, "desc")
        self.q_grasp = self._solve_ik(gp, self.q_desc, "grasp")
        self.q_carry = self._solve_carry(gp)

    def _solve_carry(self, gp):
        #LIFT的终点 = 行走携带姿态：捏点位于抓取点正上方、固定在CARRY_PINCH_Z验证过的行走抬臂手部高度这样
        #M3切割可以在此冻结抓握，走到 B 桌后放置全程零高度变化。
        cp = gp.clone()
        cp[:, 2] = CARRY_PINCH_Z
        seed = self.q_flip if self.q_tilt is None else self.q_tilt
        return self._solve_ik(cp, seed, "carry")

    def replan_desc(self):
        #FLIP 结束时按最新物体位姿重规划下探
        gp = self._grasp_point()
        self.grasp_z = gp[:, 2].clone()
        desc = self._desc_point(gp)
        seed = self.q_tilt if self.q_tilt is not None else self.q_flip
        if self.q_tilt is not None:
            hov = desc.clone()
            hov[:, 2] += self.p["hover"]
            self.q_tilt = self._solve_ik(hov, self.q_tilt, "tilt*")
            seed = self.q_tilt
        self.q_desc = self._solve_ik(desc, seed, "desc*")
        self.q_grasp = self._solve_ik(gp, self.q_desc, "grasp*")
        self.q_carry = self._solve_carry(gp)

    def plan_place(self, place_target_w):
        #规划放置腿。瞄准时使用"实测"的捏点->物体偏移放下高度 = 抓取高度 + 间隙。
        #place_reorient（drill）：航点按"静息"掌向求解——TRAV2搬运ramp随后把手腕绕机器人前向反 roll +90 度，
        #在空中把横躺的工件转成竖直；手中物体的偏移也按同样方式旋转（右->下，上->右）后再瞄准，放下高度 = 立姿根部高度。
        hover_vec = torch.zeros((1, 3), device=self.dev)
        hover_vec[:, 2] = self.p["hover"]
        right = self._right()
        if self.p.get("place_reorient", False):
            # 立式放置（drill）：把"横握时"的物体偏移旋转到"立起后"的方向
            pp_nom = (place_target_w.unsqueeze(0) + right * self.p["grasp_up"]
                      + self.fwd * self.p["grasp_fwd"])
            obj_rel = self.obj.data.root_pos_w[:1] - self.pinch_w()  # 横握状态
            c_f = (obj_rel * self.fwd).sum(dim=-1, keepdim=True)
            c_u = obj_rel[:, 2].unsqueeze(-1)
            pp = place_target_w.unsqueeze(0) - (self.fwd * c_f + right * c_u)
            pp[:, :2] = pp_nom[:, :2] + (pp - pp_nom)[:, :2].clamp(-0.05, 0.05)
            # 释放后侧移时手指会把立着的工件拖右约2cm预先向左移
            pp -= right * self.p.get("place_shove_comp", 0.0)
            # 放下高度 = 静置根高 + 立/躺根高差 + 释放间隙 - 抓取右偏
            pp[:, 2] = (self.obj_rest_z + self.p["place_root_dz"]
                        + self.p["drop_clearance"] - self.p["grasp_right"])
            # 下面的腕俯仰增量会把捏点压低约 sin(pitch)*8cm；预先抬高，使底面仍落在释放间隙高度
            pp[:, 2] += math.sin(self.p.get("place_pitch", 0.0)) * 0.09
            self.q_phover = self._solve_ik(pp + hover_vec, self.q_carry, "phover",
                                           quat_w=self.rest_quat)
            self.q_pdown = self._solve_ik(pp, self.q_phover, "pdown", quat_w=self.rest_quat)
            if self.p.get("retreat_side", 0.0) > 0.0:
                side = right * self.p["retreat_side"]
                self.q_pside = self._solve_ik(pp + side, self.q_pdown, "pside",
                                              quat_w=self.rest_quat)
                self.q_pside_up = self._solve_ik(pp + side + hover_vec, self.q_pside,
                                                 "pret", quat_w=self.rest_quat)
            if self.p.get("place_pitch", 0.0):
                # 仅动手掌的"底面调平"：指尖前倾向下。是求解"之后"的纯关节增量
                dq = torch.zeros((1, len(self.arm_ids)), device=self.dev)
                dq[0, 5] = self.p["place_pitch"]  # wrist_pitch 槽位
                self.q_pdown = self.q_pdown + dq
                if self.q_pside is not None:
                    self.q_pside = self.q_pside + dq
                    self.q_pside_up = self.q_pside_up + dq
            return
        # 普通（非立式）放置：按实测的持物偏差（钳 ±5 cm）修正瞄准点
        nominal = right * self.p["grasp_right"] + self.fwd * self.p["grasp_fwd"]
        dev = (self.pinch_w() - self.obj.data.root_pos_w[:1] - nominal)[:, :2].clamp(-0.05, 0.05)
        pp = place_target_w.unsqueeze(0) + nominal
        pp[:, :2] += dev
        pp[:, 2] = self.grasp_z + self.p["drop_clearance"]
        self.q_phover = self._solve_ik(pp + hover_vec, self.q_carry, "phover")
        self.q_pdown = self._solve_ik(pp, self.q_phover, "pdown")

    # ---- 每次查询的动作 ----
    def action28(self, tick: float, place_target_w) -> torch.Tensor:
        #给出该 tick 的原始28维BC动作
        T = EXPERT_T
        # 三次规划各自提前一个查询触发：拾取规划、FLIP末重规划、放置规划
        if not self.planned_pick and tick >= T["HOLD"] - BC_CLOCK_PER_STEP:
            self.planned_pick = True
            self.plan_pick()
        if self.planned_pick and not self.replanned and tick >= T["FLIP"] - BC_CLOCK_PER_STEP:
            self.replanned = True
            self.replan_desc()
        if self.replanned and not self.planned_place and tick >= T["HOLD2"] - BC_CLOCK_PER_STEP:
            self.planned_place = True
            self.plan_place(place_target_w)
        frac = 0.0   # 主闭合 ramp（0 = 张开/预闭合，1 = 闭合）
        pre = 0.0    # 预闭合底线（顶部进近穿孔用；侧向为 0）
        if tick < T["HOLD"] or self.q_hover is None:
            arm = self.q_start
        elif tick < T["TRAV"]:
            # 第一段运动：行走抬臂位 -> 抓取点上方悬停，掌向"不旋转"
            s = _ease((tick - T["HOLD"]) / max(1, (T["TRAV"] - T["HOLD"])))
            arm = self.q_start + s * (self.q_hover - self.q_start)
        elif tick < T["FLIP"]:
            # 悬停高度、抓取点正上方的专用翻腕步：手掌从朝左转为朝下，同时
            # 食指/中指预弯以便穿孔。侧向进近：q_flip == q_hover（沉降）。
            # grasp_pitch（drill）：窗口一分为二——先翻掌，再由 TILT 子 ramp
            # 把手掌前倾 15 度（前倾使拇指下探时不会率先扎向工件）。
            u = (tick - T["TRAV"]) / max(1, (T["FLIP"] - T["TRAV"]))
            if self.q_tilt is not None:
                # 90 tick 窗口的 5/9 = 50 tick 翻掌，40 tick 倾斜
                split = 50.0 / 90.0
                if u < split:
                    arm = self.q_hover + _ease(u / split) * (self.q_flip - self.q_hover)
                else:
                    arm = self.q_flip + _ease((u - split) / (1.0 - split)) * (
                        self.q_tilt - self.q_flip)
            else:
                arm = self.q_hover + _ease(u) * (self.q_flip - self.q_hover)
            pre = _ease(u)
        elif tick < T["DESCEND"]:
            top = self.q_tilt if self.q_tilt is not None else self.q_flip
            arm = top + _ease((tick - T["FLIP"]) / (T["DESCEND"] - T["FLIP"])) * (
                self.q_desc - top)
            pre = 1.0
        elif tick < T["INSERT"]:
            arm = self.q_desc + _ease((tick - T["DESCEND"]) / (T["INSERT"] - T["DESCEND"])) * (
                self.q_grasp - self.q_desc)
            pre = 1.0
        elif tick < T["SETTLE"]:
            arm = self.q_grasp
            pre = 1.0
        elif tick < T["CLOSE"]:
            arm = self.q_grasp
            frac, pre = (tick - T["SETTLE"]) / (T["CLOSE"] - T["SETTLE"]), 1.0
        elif tick < T["LIFT"]:
            # LIFT 直达行走携带姿态（捏点位于抓取点正上方 CARRY_PINCH_Z，抓取掌向保持）
            arm = self.q_grasp + _ease((tick - T["CLOSE"]) / (T["LIFT"] - T["CLOSE"])) * (
                self.q_carry - self.q_grasp)
            frac, pre = 1.0, 1.0
        elif tick < T["HOLD2"]:
            # 携带保持：给行走策略时间在新增负载下重新配平
            arm = self.q_carry
            frac, pre = 1.0, 1.0
        elif tick < T["TRAV2"]:
            # CARRY -> 放置目标上方。place_reorient（drill）：phover求解的目标是"静息"掌向，
            #所以这段关节ramp顺带反 roll 手腕、在空中把工件立起来。
            arm = self.q_carry + _ease((tick - T["HOLD2"]) / (T["TRAV2"] - T["HOLD2"])) * (
                self.q_phover - self.q_carry)
            frac, pre = 1.0, 1.0
        elif tick < T["LOWER"]:
            # 放下：悬停位 -> 放置位
            arm = self.q_phover + _ease((tick - T["TRAV2"]) / (T["LOWER"] - T["TRAV2"])) * (
                self.q_pdown - self.q_phover)
            frac, pre = 1.0, 1.0
        elif tick < T["RELEASE"]:
            # 手指只张回到"预闭合"姿态：孔抓取时它们仍在孔内，完全伸直会把工件推来推去
            arm = self.q_pdown
            frac, pre = 1.0 - (tick - T["LOWER"]) / (T["RELEASE"] - T["LOWER"]), 1.0
        elif tick < T["SIDE"] and self.q_pside is not None:
            # 仅立式放置：在放下高度先向机器人右侧移、再抬起
            arm = self.q_pdown + _ease((tick - T["RELEASE"]) / (T["SIDE"] - T["RELEASE"])) * (
                self.q_pside - self.q_pdown)
            pre = 1.0
        elif tick < T["RETREAT"]:
            # 撤离上升：有侧移的从侧移位上升，否则直接回悬停位
            if self.q_pside is not None:
                arm = self.q_pside + _ease((tick - T["SIDE"]) / (T["RETREAT"] - T["SIDE"])) * (
                    self.q_pside_up - self.q_pside)
            else:
                arm = self.q_pdown + _ease((tick - T["RELEASE"]) / (T["RETREAT"] - T["RELEASE"])) * (
                    self.q_phover - self.q_pdown)
            pre = 1.0
        else:
            # 冻结在撤离顶点；预闭合的手指在 0tick内松弛到完全张开。END + 短暂沉降之后，"周期后返回"把手臂 ramp回抬臂行走位
            if self.q_pside_up is not None:
                top = self.q_pside_up
            else:
                top = self.q_phover if self.q_phover is not None else self.q_start
            s = (tick - (T["END"] + BC_RETURN_SETTLE)) / BC_RETURN_TICKS
            arm = top + _ease(max(0.0, min(1.0, s))) * (self.q_start - top)
            pre = max(0.0, 1.0 - (tick - T["RETREAT"]) / 20.0)
        frac = max(0.0, min(1.0, frac))
        # 手指目标 = 预闭合底线 + 主闭合 ramp 的增量部分
        fin = self.preclose[0] * pre + (self.closed[0] - self.preclose[0]) * frac
        # 编码成原始 28 维动作：目标角 / BC_ACTION_SCALE
        act = torch.zeros((1, 28), device=self.dev)
        act[0, self.arm_slots] = arm[0] / BC_ACTION_SCALE
        act[0, self.fin_slots] = fin / BC_ACTION_SCALE
        return act


def wrap_angle(a: float) -> float:
    #把角度规范化到 (-pi, pi]。
    return math.atan2(math.sin(a), math.cos(a))


def quat_yaw(w: float, x: float, y: float, z: float) -> float:
    #从四元数 (w,x,y,z) 提取偏航角（绕 z 轴）。
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def preset_joint_gains(art_cfg, joint_name: str):
    #查询某关节在给定 ArticulationCfg 执行器配置下会得到的(stiffness, damping)
    for act in art_cfg.actuators.values():
        for expr in act.joint_names_expr:
            if re.fullmatch(expr, joint_name):
                def pick(v):
                    if isinstance(v, dict):
                        for k, val in v.items():
                            if re.fullmatch(k, joint_name):
                                return val
                        return None
                    return v
                return pick(act.stiffness), pick(act.damping)
    return None, None


def override_joint_gains(robot, jid: int, kp: float, kd: float) -> None:
    #同时在两处覆盖单个关节的 PD 增益：仿真侧（隐式执行器直接用）与执行器模型张量（显式执行器按它们算力矩）。
    robot.write_joint_stiffness_to_sim(kp, joint_ids=[jid])
    robot.write_joint_damping_to_sim(kd, joint_ids=[jid])
    for grp in robot.actuators.values():
        idx = grp.joint_indices
        if isinstance(idx, slice):
            idx = list(range(robot.num_joints))[idx]
        elif torch.is_tensor(idx):
            idx = idx.tolist()
        else:
            idx = list(idx)
        if jid in idx:
            k = idx.index(jid)
            grp.stiffness[:, k] = kp
            grp.damping[:, k] = kd


def quat_pitch_roll(w: float, x: float, y: float, z: float) -> tuple:
    #从四元数提取 (pitch, roll)，单位 rad。BC 环境的底座是"栓平"的——
    #实时的任何俯仰/横滚都会整体倾斜关节空间的手臂轨迹，所以要遥测。
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


# ============================================================
# 演示环境配置：训练的观测/动作 + "惰性"指令生成器，无奖励、无超时重置。
# ============================================================
@configclass
class DemoCommandsCfg:
    #速度指令项：与训练观测读取的同一个项类；采样被"中和"，因此我们对 vel_command_b 的写入是唯一来源。
    base_velocity = base_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=base_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
    )


@configclass
class DemoRewardsCfg:
    #奖励项：这是执行宿主、没有可优化的东西；放一个零权重项让 reward manager 走它的常规路径即可。
    alive = RewTerm(func=base_mdp.is_alive, weight=0.0)


@configclass
class DemoTerminationsCfg:
    #终止项：只保留超时（回合长度已设为 1 小时，实际上不会触发）。
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class DemoEventsCfg:
    #事件项：确定性地重置到出生位姿——不做任何随机化（演示）。
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


@configclass
class TransportLocoDemoEnvCfg(ManagerBasedRLEnvCfg):
    #在双桌搬运场景里宿主行走策略的环境配置。

    scene: TransportSceneCfg = TransportSceneCfg(num_envs=1, env_spacing=20.0)
    observations: ObservationsCfg = ObservationsCfg()   # 训练的观测布局，原样复用
    actions: ActionsCfg = ActionsCfg()                  # 训练的动作映射，原样复用
    commands: DemoCommandsCfg = DemoCommandsCfg()
    rewards: DemoRewardsCfg = DemoRewardsCfg()
    terminations: DemoTerminationsCfg = DemoTerminationsCfg()
    events: DemoEventsCfg = DemoEventsCfg()
    curriculum = None

    def __post_init__(self):
        # 浮动底座机器人放在搬运出生位
        self.scene.robot = G1RobotPresets.g1_29dof_dex3_wholebody(
            init_pos=(ROBOT_START_POS[0], ROBOT_START_POS[1], 0.80),
            init_rot=ROBOT_START_ROT,
        )
        self.decimation = 4                 # 50 Hz 控制，与训练一致
        self.episode_length_s = 3600.0      # 演示途中绝不超时
        self.sim.dt = 0.005                 # 物理步长 5 ms
        # --render_skip N 每 N 个控制帧才画一帧：纯显示节流，物理/控制步进逐字节相同。
        self.sim.render_interval = self.decimation * max(1, args_cli.render_skip)
        if args_cli.record:
            # 两个固定调试相机。两张桌子是同一个USD 资产，故A桌（旋转了 -90 度）沿x的半"深"为 0.37
            #其西侧中腿在 (-0.37, 0)。高度 = 桌面 + 1 m，光轴低于水平 45 度。
            cam_z = TABLETOP_Z + CAM_ABOVE_TOP
            # A 桌相机：西侧中腿，朝东看整排拾取行。焦距极短：7 行工件跨
            # y ±0.81 而距离只有约 0.5 m，需要约 120 度水平视场才能覆盖两端。
            self.scene.cam_table_a = CameraCfg(
                prim_path="/World/envs/env_.*/CamTableA",
                update_period=CTRL_DT,
                width=REC_RES[0], height=REC_RES[1],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=6.0, clipping_range=(0.05, 30.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=(-TABLE_B_HALF_DEPTH, 0.0, cam_z),
                    rot=_cam_quat(0.0, CAM_PITCH_DOWN_DEG),
                    convention="world"),
            )
            # B 桌相机：南侧中腿，朝北看、向东偏 CAM_B_EAST_BIAS 度，好把螺母的东侧放置点收进画面。
            self.scene.cam_table_b = CameraCfg(
                prim_path="/World/envs/env_.*/CamTableB",
                update_period=CTRL_DT,
                width=REC_RES[0], height=REC_RES[1],
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=10.0, clipping_range=(0.05, 30.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=(TABLE_B_POS[0], TABLE_B_POS[1] - TABLE_B_HALF_DEPTH, cam_z),
                    rot=_cam_quat(90.0 - CAM_B_EAST_BIAS_DEG, CAM_PITCH_DOWN_DEG),
                    convention="world"),
            )


@configclass
class FlatLocoDemoEnvCfg(TransportLocoDemoEnvCfg):
    #A/B 隔离变体：训练场景，演示机制完全相同。

    def __post_init__(self):
        super().__post_init__()
        self.scene = LocoSceneCfg(num_envs=1, env_spacing=20.0)
        #（LocoSceneCfg 的机器人出生在原点、偏航为单位角；它的接触传感器跟着一起加载但不使用。）
        self.scene.contact_forces.prim_path = self.scene.robot.prim_path + "/.*"


class WalkPolicy(torch.nn.Module):
    #skrl共享高斯策略的"确定性"重实例化。

    #按skrl的模块命名镜像yaml网络结构（[256,128,128]+ELU，共享价值头），使checkpoint的`policy`state_dict 能严格加载。
    #同时复刻RunningStandardScaler 状态预处理器
    
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # 主干：obs -> 256 -> 128 -> 128，ELU 激活（对应训练 yaml）
        self.net_container = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 128), torch.nn.ELU(),
        )
        self.policy_layer = torch.nn.Linear(128, act_dim)
        self.value_layer = torch.nn.Linear(128, 1)      # 不使用；只为严格加载而存在
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(act_dim))
        # 观测归一化的运行均值/方差（从 checkpoint 的预处理器状态填充）
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var", torch.ones(obs_dim))

    @staticmethod
    def from_checkpoint(path: str, device: str) -> "WalkPolicy":
        #从 skrl checkpoint 构建推理用策略（维度从权重形状推断）。
        ckpt = torch.load(path, map_location=device, weights_only=False)
        sd = ckpt["policy"]
        obs_dim = sd["net_container.0.weight"].shape[1]
        act_dim = sd["policy_layer.weight"].shape[0]
        model = WalkPolicy(obs_dim, act_dim).to(device)
        # strict=False只是因为我们额外加了 scaler 缓冲区；
        missing, unexpected = model.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not k.startswith("obs_")]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint/model mismatch: missing={missing} unexpected={unexpected}")
        pre = ckpt.get("state_preprocessor", None)
        if pre and "running_mean" in pre:
            model.obs_mean[:] = pre["running_mean"].to(device).flatten()
            model.obs_var[:] = pre["running_variance"].to(device).flatten()
            print(f"[demo] state preprocessor loaded (mean |.|={model.obs_mean.abs().mean():.3f})")
        else:
            print("[demo] WARNING: no state_preprocessor in checkpoint - was it trained with one?")
        model.eval()
        print(f"[demo] walk policy loaded: obs={obs_dim}, act={act_dim} <- {path}")
        return model

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # skrl RunningStandardScaler的推理变换（±5 个标准差截断）。
        x = (obs - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-8)
        x = torch.clamp(x, -5.0, 5.0)
        return self.policy_layer(self.net_container(x))


def main() -> None:
    #演示主函数：构建环境 -> 加载策略 -> 主控制循环（状态机 + BC）
    # --- 环境选择与开关的组合校验 ---
    if args_cli.scene == "flat":
        if args_cli.target:
            parser.error("--target is a transport-scene feature (flat has no workpieces)")
        env_cfg = FlatLocoDemoEnvCfg()
        goal_xy, goal_yaw = FLAT_GOAL_XY, FLAT_GOAL_YAW
    else:
        env_cfg = TransportLocoDemoEnvCfg()
        if args_cli.target and args_cli.no_objects:
            parser.error("--target needs the workpieces; drop --no_objects")
        if args_cli.target and args_cli.dock != "A":
            parser.error("--target only defines table-A pick docks (dock B carries the piece in hand)")
        if args_cli.bc_checkpoint and not args_cli.target:
            parser.error("--bc_checkpoint needs --target (which piece the BC picks; must match its training)")
        if args_cli.bc_expert and not args_cli.target:
            parser.error("--bc_expert needs --target (which piece the scripted expert picks)")
        if args_cli.bc_dagger and not args_cli.bc_checkpoint:
            parser.error("--bc_dagger runs the NET with expert labels; add --bc_checkpoint")
        if args_cli.bc_dagger and args_cli.bc_expert:
            parser.error("--bc_dagger and --bc_expert are mutually exclusive (net vs expert execution)")
        if args_cli.bc_dagger and not args_cli.bc_raised:
            # 重采数据集（以及在其上训练的网络）都从行走抬臂位开始；
            # DAgger 必须以同样方式进入
            args_cli.bc_raised = True
            print("[demo] --bc_dagger implies --bc_raised")
        if args_cli.loop_all:
            missing = [f"--bc_{c}" for c in ("cube", "bolt", "nut", "drill")
                       if not getattr(args_cli, f"bc_{c}")]
            if missing:
                parser.error(f"--loop_all needs a checkpoint per class: {' '.join(missing)}")
            for bad in ("target", "bc_checkpoint", "only_target", "bc_expert", "bc_dagger",
                        "save_dataset", "dump_obs", "spawn_at_dock", "bc_start_tick",
                        "bc_statue", "bc_immediate"):
                if getattr(args_cli, bad):
                    parser.error(f"--loop_all is a self-contained NET mode; drop --{bad}")
            if args_cli.place_zone is not None:
                parser.error("--loop_all derives every place spot itself; drop --place_zone")
            # 循环 = 7个串联的 --carry_to_b 搬运；任务 1 先播种共享的单目
            # 标管线，switch_to_job 再逐任务重新瞄准
            args_cli.carry_to_b = True
            args_cli.target = M4_JOBS[0][1]
            args_cli.bc_checkpoint = args_cli.bc_cube
            if args_cli.repeat < 1:
                parser.error("--repeat must be >= 1")
            if args_cli.max_time == 90.0:
                args_cli.max_time = 1800.0
                print("[demo] --loop_all: --max_time raised to 1800 s (7 transports)")
        elif args_cli.jobs:
            parser.error("--jobs filters the --loop_all job table; add --loop_all")
        elif args_cli.repeat != 1:
            parser.error("--repeat batches --loop_all iterations; add --loop_all")
        if args_cli.carry_to_b:
            if not args_cli.bc_checkpoint:
                parser.error("--carry_to_b runs the NET place leg at table B; add --bc_checkpoint")
            if args_cli.bc_expert or args_cli.bc_dagger:
                parser.error("--carry_to_b is NET-only for now (no --bc_expert / --bc_dagger)")
            if args_cli.save_dataset:
                parser.error("--carry_to_b does not record datasets")
            if args_cli.bc_statue or args_cli.bc_immediate:
                parser.error("--carry_to_b needs the walking base and the raised-start entry")
            if not args_cli.bc_raised:
                args_cli.bc_raised = True
                print("[demo] --carry_to_b implies --bc_raised")
            if args_cli.max_time == 90.0:
                # 默认预算只够 A 桌那条腿；约 6 m 走到 B 桌 + 停靠 + 放置腿需要更多时间
                args_cli.max_time = 240.0
                print("[demo] --carry_to_b: --max_time raised to 240 s (walk to table B included)")
        if args_cli.save_dataset and not (args_cli.bc_expert or args_cli.bc_dagger):
            parser.error("--save_dataset records expert-labelled rows; add --bc_expert or --bc_dagger")
        if args_cli.action_noise > 0.0 and not (args_cli.bc_expert or args_cli.bc_dagger):
            parser.error("--action_noise perturbs the executed action; add --bc_expert or --bc_dagger")
        if args_cli.no_objects:
            # 实例级置 None（world_camera 同款技巧）：直接不生成这些 prim。
            for ent in ALL_WORKPIECE_ENTITIES:
                setattr(env_cfg.scene, ent, None)
        if args_cli.only_target:
            if not args_cli.target:
                parser.error("--only_target needs --target (which piece to keep)")
            # 单工件采集场景：环境相同，其余工件全部丢弃
            for ent in ALL_WORKPIECE_ENTITIES:
                if ent != WORKPIECE_ENTITY[args_cli.target]:
                    setattr(env_cfg.scene, ent, None)
        # 静态停靠位：A 或 B0..B2（--target 的动态停靠稍后会覆盖 goal_xy）
        if args_cli.dock == "A":
            dock_pos, dock_rot = DOCK_A_POS, DOCK_A_ROT
        else:
            dock_pos, dock_rot = DOCK_B_POSES[int(args_cli.dock[1])], DOCK_B_ROT
        goal_xy, goal_yaw = (dock_pos[0], dock_pos[1]), quat_yaw(*dock_rot)
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    # 构建真正的 ManagerBasedRLEnv（v2 架构的核心：观测/动作管线与训练一致）
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    robot = env.scene["robot"]
    cmd_term = env.command_manager.get_term("base_velocity")

    policy = WalkPolicy.from_checkpoint(args_cli.walk_checkpoint, device)

    # ---- M2：BC 抓放策略（可选）----
    def load_bc_policy(path: str, tag: str):
        #加载一个 bc_train.py 训出的 110->28 BC 策略并做维度断言。
        from bc_train import SharedPolicy
        bc_ckpt = torch.load(path, map_location=device)
        assert bc_ckpt["obs_dim"] == 110 and bc_ckpt["act_dim"] == 28, (
            f"unexpected BC dims obs={bc_ckpt['obs_dim']} act={bc_ckpt['act_dim']} (want 110/28)")
        pol = SharedPolicy(bc_ckpt["obs_dim"], bc_ckpt["act_dim"]).to(device)
        pol.load_state_dict(bc_ckpt["model"])
        pol.eval()
        print(f"[demo] BC policy loaded: {path} "
              f"(val MSE {bc_ckpt.get('val_mse', float('nan')):.6f}) - target {tag}")
        return pol

    bc_policy = None
    bc_policies = {}
    if args_cli.loop_all:
        # M4：四个类别各加载一个 checkpoint，switch_to_job 时切换
        bc_policies = {c: load_bc_policy(getattr(args_cli, f"bc_{c}"), c)
                       for c in ("cube", "bolt", "nut", "drill")}
        bc_policy = bc_policies[args_cli.target]
    elif args_cli.bc_checkpoint:
        bc_policy = load_bc_policy(args_cli.bc_checkpoint, args_cli.target)
    # 脚本化专家可以替代（或叠加于）网络驱动 M2 周期；两者任一都会激活整套 BC 管线/交接机制
    bc_active = bc_policy is not None or args_cli.bc_expert

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]
    # 观测维度必须与checkpoint一致（69 维），否则说明配置不匹配
    assert obs.shape[-1] == policy.obs_mean.numel(), (
        f"env obs dim {obs.shape[-1]} != checkpoint obs dim {policy.obs_mean.numel()}")

    # ---- 采集模式的出生抖动（仅位置，且在动态停靠读取工件位姿"之前"，
    # 停靠因此会重新瞄准抖动后的位置；工件在行走/STAND 保持期间重新静置）----
    def apply_spawn_jitter() -> None:
        #给每个相关工件加一个新的随机 (x, y) 偏移。--repeat 每轮环境重置后也会调用。
        if not (args_cli.target and args_cli.spawn_jitter > 0.0):
            return
        # 循环模式：每个工件各自抖动（动态停靠逐任务重新瞄准）；
        # 单目标模式：只抖目标工件
        jit_ents = ([ent for ent, _ in M4_JOBS] if args_cli.loop_all
                    else [WORKPIECE_ENTITY[args_cli.target]])
        for jit_ent in jit_ents:
            jit_obj = env.scene[jit_ent]
            jit_pose = jit_obj.data.root_state_w[0:1, :7].clone()
            jx = random.uniform(-args_cli.spawn_jitter, args_cli.spawn_jitter)
            jy = random.uniform(-args_cli.spawn_jitter, args_cli.spawn_jitter)
            jit_pose[0, 0] += jx
            jit_pose[0, 1] += jy
            jit_obj.write_root_pose_to_sim(jit_pose)
            jit_obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=jit_pose.device))
            print(f"[demo] spawn jitter: {jit_ent} shifted ({jx:+.3f}, {jy:+.3f}) m")

    apply_spawn_jitter()

    # ---- 动态A桌停靠：按目标工件的"实时"位置计算骨盆目标，复现该工件
    # BC数据集采集时的相对位姿：前方 (0.33 + BC_SPAWN_FWD)、机器人右侧 (dock_right + BC_SPAWN_RIGHT)
    # 逐工件偏移是承重的：螺母的 BC 位姿是前 0.53 / 右 0.15（grasp-expert 第 17 轮把它挪出静息手位），
    # 停靠位 A 面向-x（偏航 180 度），所以"前方" = 世界 -x、"机器人右侧" =世界 +y：
    #   pelvis_x = piece_x + ahead, pelvis_y = piece_y - right
   
    target_obj = None
    dock_ahead = dock_right = 0.0
    if args_cli.target:
        target_obj = env.scene[WORKPIECE_ENTITY[args_cli.target]]
        obj_xy = target_obj.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]
        
        dock_ahead = PICK_DOCK_AHEAD + BC_SPAWN_FWD[args_cli.target]

        if args_cli.target == "nut":
            dock_right = args_cli.nut_dock_right
        elif args_cli.target == "drill":
            dock_right = args_cli.drill_dock_right
        else:
            dock_right = args_cli.dock_right + BC_SPAWN_RIGHT[args_cli.target]
        # dock_ahead/dock_right = "训练"的工件位姿；伺服要按落点偏差修正
        aim_ahead = dock_ahead + DOCK_AHEAD_TRIM_PICK.get(args_cli.target, DOCK_AHEAD_TRIM)
        aim_right = dock_right + DOCK_RIGHT_TRIM
        goal_xy = (obj_xy[0].item() + aim_ahead, obj_xy[1].item() - aim_right)
        print(f"[demo] target {args_cli.target} at xy=({obj_xy[0]:.2f}, {obj_xy[1]:.2f}) "
              f"-> dynamic dock A, BC-relative pose: piece {dock_ahead:.2f} m ahead / "
              f"{dock_right:.2f} m right of pelvis (servo aims {aim_ahead:.2f}/{aim_right:.2f})")
    # ---- M3 放置腿的停靠几何（--carry_to_b）----
    # 训练时放置目标位于 工件 + (f_off 前, r_off 右)，而工件位于(dock_ahead, dock_right)——因此在 B 桌，"区域中心"必须落在相对骨盆
    # (dock_ahead + f_off, dock_right + r_off) 处，偏移取采样盒中点与 A 桌相同的动态停靠数学、相同
    # 的落点偏差修正、相同的逐轴 M1 门限——只是参考对象从工件换成了区域。
    m3_zone = None
    m3_place_desc = ""
    place_zone_xy = None
    place_ahead = place_right = 0.0
    aim_ahead_b = aim_right_b = 0.0
    m3_dock_yaw = quat_yaw(*DOCK_B_ROT)
    m3_route = []
    m3_cut = 0
    if args_cli.carry_to_b:
        m3_cut = M3_CUT_TICK[args_cli.target]
        if args_cli.target == "nut" and args_cli.place_zone is None:
            # 螺母：东侧短边中点，从 +x 端进近停靠
            place_zone_xy = M3_NUT_PLACE_XY
            m3_dock_yaw = M3_NUT_DOCK_YAW
            m3_place_desc = "east-side spot"
        else:
            m3_zone = (args_cli.place_zone if args_cli.place_zone is not None
                       else M3_PLACE_ZONE[args_cli.target])
            place_zone_xy = (PLACE_ZONE_XS[m3_zone], PLACE_ZONE_Y)
            m3_place_desc = f"zone {m3_zone}"
        r_lo, r_hi, f_lo, f_hi = BC_PLACE_BOXES[args_cli.target]
        place_ahead = dock_ahead + 0.5 * (f_lo + f_hi)
        place_right = dock_right + 0.5 * (r_lo + r_hi)
        aim_ahead_b = place_ahead + DOCK_AHEAD_TRIM_PLACE.get(args_cli.target, DOCK_AHEAD_TRIM)
        aim_right_b = (place_right + DOCK_RIGHT_TRIM
                       + DOCK_RIGHT_TRIM_PLACE.get(args_cli.target, 0.0))
        # 名义朝向下的静态停靠目标；路线终点是放置点自身轴线上的整备点，螺母另加北侧走廊绕行——见m3_carry_route_for
        fbx, fby = math.cos(m3_dock_yaw), math.sin(m3_dock_yaw)
        rbx, rby = math.sin(m3_dock_yaw), -math.cos(m3_dock_yaw)
        dock_goal_b = (place_zone_xy[0] - aim_ahead_b * fbx - aim_right_b * rbx,
                       place_zone_xy[1] - aim_ahead_b * fby - aim_right_b * rby)
        m3_route = m3_carry_route_for(
            place_zone_xy, m3_dock_yaw, aim_ahead_b,
            args_cli.target == "nut" and args_cli.place_zone is None)
        route_txt = " -> ".join(f"({x:.2f},{y:.2f})" for x, y in m3_route)
        print(f"[demo] M3 carry-to-B armed: place {m3_place_desc} centre "
              f"xy=({place_zone_xy[0]:.2f}, {place_zone_xy[1]:.2f}), dock yaw "
              f"{math.degrees(m3_dock_yaw):.0f} deg puts it {place_ahead:.3f} m ahead / "
              f"{place_right:.3f} m right of the pelvis (training place-target pose); "
              f"route {route_txt} -> dock ({dock_goal_b[0]:.2f},{dock_goal_b[1]:.2f}); "
              f"cut at bc t={m3_cut}")
    print(f"[demo] goal: xy=({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) yaw={math.degrees(goal_yaw):.0f} deg")

    # ---- 快速动作调参模式：传送到停靠位（--spawn_at_dock）----
    if args_cli.spawn_at_dock:
        root = robot.data.root_state_w[0:1].clone()
        root[0, 0] = env.scene.env_origins[0, 0] + goal_xy[0]
        root[0, 1] = env.scene.env_origins[0, 1] + goal_xy[1]
        # 保留出生高度；面向停靠朝向
        root[0, 3] = math.cos(goal_yaw / 2.0)
        root[0, 4] = 0.0
        root[0, 5] = 0.0
        root[0, 6] = math.sin(goal_yaw / 2.0)
        root[0, 7:13] = 0.0     # 速度清零
        robot.write_root_pose_to_sim(root[:, :7])
        robot.write_root_velocity_to_sim(root[:, 7:13])
        print(f"[demo] SPAWN AT DOCK: teleported to xy=({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) "
              f"yaw={math.degrees(goal_yaw):.0f} deg - skipping TURN/NAV/ALIGN; STAND "
              f"stabilises for {args_cli.stand_time:.0f}s and the M1 gate still scores")

    # ---- 右臂保持管线（这些 PD 目标在行走策略的关节集之外）----
    palm_id = robot.find_bodies(PALM_LINK)[0][0]
    arm_ids, arm_names = robot.find_joints(["right_shoulder_.*", "right_elbow_joint", "right_wrist_.*"])
    q_arm_default = robot.data.default_joint_pos[:, arm_ids].clone()
    # 抬臂行走位：wholebody 默认位 + 肘部向上折起
    elbow_slot = arm_names.index("right_elbow_joint")
    wristp_slot = arm_names.index("right_wrist_pitch_joint")
    q_arm_walk = q_arm_default.clone()
    q_arm_walk[:, elbow_slot] += args_cli.raise_elbow
    q_arm_walk[:, wristp_slot] += args_cli.raise_wrist_pitch
    elbow_jid = arm_ids[elbow_slot]         # 供 tgt/act 遥测
    # BC 起始位：右臂所有关节为 0
    q_arm_bc = torch.zeros_like(q_arm_default)
    raise_frac = 0.0                    # 0 = wholebody 默认位，1 = 抬臂行走位
    bc_frac = 0.0                       # 0 = 行走位，1 = BC 起始位
    done_wait = 0.0                     # DONE 内下放 ramp 结束后的等待时间
    raise_step = CTRL_DT / RAISE_RAMP_TIME  # 每控制帧 raise_frac 的增量
    bc_step = CTRL_DT / BC_RAMP_TIME        # 每控制帧 bc_frac 的增量
    # 符号标定打印：每个关节的默认位 / 软限位 / 抬臂目标
    joint_limits = getattr(robot.data, "joint_pos_limits", None)
    if joint_limits is None:                       # 改名前的旧版 Isaac Lab
        joint_limits = robot.data.joint_limits
    for jid, jname in zip(arm_ids, arm_names):
        d = robot.data.default_joint_pos[0, jid].item()
        lo, hi = joint_limits[0, jid].tolist()
        tgt = f" walk_target={d + args_cli.raise_elbow:+.2f}" if jname == "right_elbow_joint" else ""
        print(f"[demo] arm joint {jname}: default={d:+.2f} limits=[{lo:+.2f}, {hi:+.2f}]{tgt}")

    # ---- M2：BC 交接管线 ----
    expert = None
    expert_rows = None
    if bc_active:
        # 观测下标假设 43 关节资产（29 身体 + 14 Dex3），先断言
        assert robot.num_joints == 43, (
            f"asset has {robot.num_joints} joints, BC obs indices assume 43 (29 body + 14 Dex3)")
        bc_act_ids, bc_act_names = robot.find_joints(BC_ACTION_JOINT_NAMES)  # 资产顺序，与训练一致
        assert len(bc_act_ids) == 28, f"resolved {len(bc_act_ids)}/28 BC action joints"
        # 行走策略"拥有"7 个左臂关节。BC 只写其余 21 个：右臂 7 + 双Dex3 手 14。
        bc_write_slots = [i for i, n in enumerate(bc_act_names)
                          if not re.match(r"left_(shoulder|elbow|wrist)", n)]
        bc_write_ids = [bc_act_ids[i] for i in bc_write_slots]
        bc_write_slots_t = torch.tensor(bc_write_slots, device=device, dtype=torch.long)
        # 肘在"被写关节"中的位置，供 BC 初期的 tgt-vs-act 遥测
        bc_elb_write_idx = bc_write_ids.index(elbow_jid)
        bc_body_idx = torch.tensor(BC_BODY_OBS_IDX, device=device, dtype=torch.long)
        bc_hand_idx = torch.tensor(BC_HAND_OBS_IDX, device=device, dtype=torch.long)
        bc_frozen_ids, _ = robot.find_joints(BC_FROZEN_JOINT_PATTERNS)
        bc_left_ids, bc_left_names = robot.find_joints(list(BC_LEFT_ARM_OBS_POSE.keys()))
        bc_left_pose = torch.tensor([BC_LEFT_ARM_OBS_POSE[n] for n in bc_left_names], device=device)
        lw_id = robot.find_bodies(BC_HAND_LINKS["left"])[0][0]
        rw_id = robot.find_bodies(BC_HAND_LINKS["right"])[0][0]
        bc_last_action = torch.zeros(1, 28, device=device)
        bc_clock = 0.0                  # BC 步计数（100 Hz 语义）
        bc_place_target_w = None        # 放置目标（世界系），交接时采样
        bc_obj_rest_z = 0.0             # 工件静置高度（抬升量的基准）
        bc_obj_max_z = 0.0              # BC 期间物体高度峰值：区分真正的抓起+抬升(bc_play：+9.6 cm)与把工件沿桌面推进目标
        bc_settle_elapsed = 0.0         # 到位状态已保持的时间
        bc_report_done = False          # M2 报告是否已输出
        bc_q_from = None                # 交接时捕获的被写关节姿态；
        bc_blend_frac = 0.0             # 即时接管的混合起点/进度
        # --- 给 bc_obs_diff.py 的观测转储 ---
        bc_dump_rows = [] if args_cli.dump_obs else None
        bc_dump_saved = False
        bc_handover_meta = None
        # --- 训练参考热启动（见 --bc_start_tick / --bc_ref_dump）---
        bc_ref = None
        if args_cli.bc_ref_dump:
            bc_ref = torch.load(args_cli.bc_ref_dump, map_location="cpu", weights_only=False)
            print(f"[demo] BC reference dump loaded: {args_cli.bc_ref_dump} "
                  f"({bc_ref['obs'].shape[0]} rows, workpiece {bc_ref.get('workpiece')})")
        if args_cli.bc_start_tick > 0:
            if bc_ref is not None:
                k = args_cli.bc_start_tick
                # 训练网络在 tick k 看到的是"它自己"tick k-1 的动作，不是零
                bc_last_action = bc_ref["act"][k - 1].to(device).unsqueeze(0).clone()
                # 把 DONE 下放改指训练tick k 的右臂姿态，例如 t=20 时肘 ~-0.05，而非全零的 t=0 位
                ref_jp = bc_ref["obs"][k, :29]
                ref_names = bc_ref["body_names"]
                for slot, name in enumerate(arm_names):
                    if name in ref_names:
                        q_arm_bc[0, slot] = ref_jp[ref_names.index(name)].item()
                print(f"[demo] BC warm start at tick {k}: last_action seeded from tick {k - 1}, "
                      f"arm descent retargeted to the trained tick-{k} pose "
                      f"(elbow {q_arm_bc[0, elbow_slot].item():+.3f})")
            else:
                print(f"[demo] WARNING: --bc_start_tick {args_cli.bc_start_tick} without "
                      f"--bc_ref_dump: last_action starts at ZERO and the arm descends to "
                      f"the t=0 pose - state/phase/feedback are NOT fully aligned")
        # 雕像模式：行走动作项的 19 个关节，解析方式与动作项本身完全一致；
        # 交接时合成能 PD 保持所捕获站姿的"常量动作"
        walk_act_ids, _ = robot.find_joints(WALK_ACTION_JOINT_NAMES)
        bc_freeze_action = None

        stiff = getattr(robot.data, "joint_stiffness", None)
        damp = getattr(robot.data, "joint_damping", None)
        if stiff is None:                               # 旧版 Isaac Lab 的命名
            stiff = getattr(robot.data, "default_joint_stiffness", None)
            damp = getattr(robot.data, "default_joint_damping", None)
        # 参考配置：base_fix预设（BC 数据采集时的增益来源）
        bc_ref_cfg = G1RobotPresets.g1_29dof_dex3_base_fix(
            init_pos=(0.0, 0.0, 0.76), init_rot=(1.0, 0.0, 0.0, 0.0))
        bc_gain_plan = []                       # (jid, kp0, kd0, kp1, kd1)：软 -> base_fix
        grip_gain_refs = []                     # 右手手指：(jid, kp_ref, kd_ref)
        for slot in bc_write_slots:
            jid, jname = bc_act_ids[slot], bc_act_names[slot]
            kp_ref, kd_ref = preset_joint_gains(bc_ref_cfg, jname)
            kp_now = stiff[0, jid].item() if stiff is not None else float("nan")
            kd_now = damp[0, jid].item() if damp is not None else float("nan")
            if kp_ref is None or kd_ref is None:
                print(f"[demo] BC-joint PD {jname}: kp={kp_now:7.2f} kd={kd_now:6.3f} "
                      f"(base_fix value USD-driven, NOT overridden)")
                continue
            # 读不到当前值（旧 API）-> 没有 ramp 起点：直接硬应用
            kp0 = float(kp_ref) if math.isnan(kp_now) else float(kp_now)
            kd0 = float(kd_ref) if math.isnan(kd_now) else float(kd_now)
            bc_gain_plan.append((jid, kp0, kd0, float(kp_ref), float(kd_ref)))
            if jname.startswith("right_hand_"):
                # 搬运握紧的目标增益（见 M3_CARRY_GRIP_KP_SCALE）
                grip_gain_refs.append((jid, float(kp_ref), float(kd_ref)))
            print(f"[demo] BC-joint PD {jname}: kp {kp_now:7.2f} -> {kp_ref:7.2f} "
                  f"kd {kd_now:6.3f} -> {kd_ref:6.3f} (base_fix, ramped over "
                  f"{GAIN_RAMP_TIME:.1f}s at handover)")
        # 脚本化专家（--bc_expert / --bc_dagger）：只有在EXPERT_PROFILES里登记过抓取几何的工件类别才能用
        if args_cli.bc_expert or args_cli.bc_dagger:
            prof = EXPERT_PROFILES.get(args_cli.target)
            if prof is None:
                raise SystemExit(
                    f"--bc_expert/--bc_dagger support {sorted(EXPERT_PROFILES)}; "
                    f"'{args_cli.target}' has no profile")
            expert = ScriptedExpert(robot, target_obj, device, prof, bc_act_names)
            expert_rows = [] if args_cli.save_dataset else None
            role = ("DAgger LABELLER (the net drives)" if args_cli.bc_dagger
                    else "plan-then-execute on the standing base")
            print(f"[demo] scripted expert armed for '{args_cli.target}' "
                  f"({prof.get('approach', 'side').upper()}-approach RAISED-START {role}"
                  + (f", recording dataset -> {args_cli.save_dataset}" if args_cli.save_dataset else "")
                  + ")")

    # ================== 状态机初始状态 ==================
    phase = "TURN"                  # 状态机起始阶段：先原地转向面对目标
    if args_cli.spawn_at_dock:
        phase = "STAND"             # 直接出生在停靠点 -> 跳过导航，从 STAND 开始
        print(f"[demo] starting in STAND at the dock")
    # M3 腿程追踪器："pick" -> "carry" -> "place"
    m3_leg = "pick"
    m3_carry_hold = False           # True 表示冻结的抓取目标"不许"被常规的手臂保持写入覆盖
    m3_wp = 0                       # m3_route 中下一个航点的下标

    m3_backoff_dist = M3_BACKOFF_DIST   # 按腿程可变：M4 返回腿要一直退到撤离走廊（见 ARMRET）
    
    backoff_start_xy = None         # 后退撤离的起点（用来量已退距离）
    backoff_elapsed = 0.0           # 后退阶段已耗时（超时保护）
    stand_elapsed = 0.0             # STAND 阶段已站立时间
    align_elapsed = 0.0             # ALIGN 阶段累计耗时（ALIGN_MAX_SEC 超时用）
    align_yaw_ref, align_stuck = None, 0.0   # 偏航死锁检测：参考偏航 + 卡住计时
    retry_elapsed, retry_count = 0.0, 0      # RETRY 后退重试的计时与次数
    bc_gain_frac = None                 # None = 增益 ramp 未开始（STAND 交接时启动）
    gain_wait = 0.0                     # 抬臂入场：保持抬臂等增益 ramp 完成的时间
    stand_start_xy = (goal_xy[0], goal_xy[1]) if args_cli.spawn_at_dock else None
    t = 0.0                             # 仿真累计时间（秒）
    last_report = -1.0                  # 上次打印遥测的时间（限频用）

    # ---- M4（--loop_all）：作业表 + 逐作业重新瞄准 ----
    cur_cls = args_cli.target           # "当前作业"的工件类别
    m4_jobs = []
    if args_cli.loop_all:
        # 按 M4_JOBS 的固定顺序展开成作业记录：每条含场景实体名、类别、 B 桌放置点坐标、停靠偏航、人类可读描述
        seen = {}                       # 同类别已出现次数（决定东/西点位）
        for ent, cls in M4_JOBS:
            k = seen.get(cls, 0)
            seen[cls] = k + 1
            if cls == "nut":
                # 螺母走 B 桌东侧的专用点与专用停靠朝向
                spot, dyaw, desc = M3_NUT_PLACE_XY, M3_NUT_DOCK_YAW, "east-side spot"
            else:
                # 其余类别按 M3_PLACE_ZONE 分配放置区；同类第 1 件放东点、第 2 件放西点（相距 2*PLACE_SPOT_DX）
                z = M3_PLACE_ZONE[cls]
                dx = PLACE_SPOT_DX if k == 0 else -PLACE_SPOT_DX
                spot = (PLACE_ZONE_XS[z] + dx, PLACE_ZONE_Y)
                dyaw = quat_yaw(*DOCK_B_ROT)
                desc = f"zone {z} {'east' if k == 0 else 'west'} spot"
            m4_jobs.append({"entity": ent, "cls": cls, "spot": spot,
                            "dock_yaw": dyaw, "desc": desc})
        if args_cli.jobs:
            # --jobs nut/--jobs 4,7：只跑子集点位保持"完整布局"下的分配不变，因此比如作业7仍然放到2号区的西点，即使作业4根本没跑。
            toks = [tk.strip().lower() for tk in args_cli.jobs.split(",") if tk.strip()]
            valid = ({str(i + 1) for i in range(len(m4_jobs))}
                     | {jb["cls"] for jb in m4_jobs})
            bad = [tk for tk in toks if tk not in valid]
            if bad:
                raise SystemExit(f"[demo] --jobs: unknown token(s) {bad} - use job numbers "
                                 f"1-{len(m4_jobs)} and/or class names cube/bolt/nut/drill")
            m4_jobs = [jb for i, jb in enumerate(m4_jobs)
                       if str(i + 1) in toks or jb["cls"] in toks]
            picked = ", ".join(f"{jb['cls']} -> {jb['desc']}" for jb in m4_jobs)
            print(f"[demo] --jobs {args_cli.jobs}: running {len(m4_jobs)} of "
                  f"{len(M4_JOBS)} jobs ({picked})")
    m4_job_idx = 0
    m4_results = []                     # 每个完成的作业留一行判定结论
    m4_done = False
    m4_dropped = False                  # 本次作业搬运途中掉件
    armret_frac = 0.0                   # 放置结束 -> 行走抬臂位 的手臂 ramp 进度
    armret_from = None
    # ---- --repeat 批量统计状态 ----
    # m4_records每行：(作业序号, 类别, 点位描述, 判定, 物体->目标距离cm或None,停靠重试次数)是 m4_results 字符串的结构化孪生，在同样三个
    # 位置追加（M2 判定、搬运途中掉件、看门狗超时），最终由 m4_batch_summary() 汇总成逐作业成功率。
    m4_records = []
    m4_iter = 1                         # 当前迭代（从 1 计数）
    m4_iter_end = False                 # 本迭代"最后一个"作业出分时置位
    m4_summary_done = False
    iter_t0 = 0.0                       # 迭代开始时刻的仿真时间
    # 单次迭代的时间上限 = 每作业上限 x 作业数
    m4_iter_cap = args_cli.iter_cap * max(1, len(m4_jobs))
    m4_stats_csv = None
    if args_cli.loop_all and args_cli.repeat > 1:
        # 自动把 --max_time 抬高到足够跑完全部迭代（外加 120 s 余量）
        need = args_cli.repeat * m4_iter_cap + 120.0
        if args_cli.max_time < need:
            args_cli.max_time = need
            print(f"[demo] --repeat {args_cli.repeat}: --max_time raised to {need:.0f} s "
                  f"({args_cli.repeat} iterations x {len(m4_jobs)} job(s) x "
                  f"{args_cli.iter_cap:.0f} s cap)")
        m4_stats_csv = (args_cli.stats_csv
                        or f"m4_stats_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        print(f"[demo] --repeat: per-iteration verdicts stream to {m4_stats_csv}")

    def pick_dock_geom(cls: str):
        #返回某个工件类别在拾取侧的"训练时工件相对停靠偏移"（前方距离、右侧距离）
        # 前方距离 = 基础停靠距离 + 该类别 BC 训练时的出生前向偏移
        d_ahead = PICK_DOCK_AHEAD + BC_SPAWN_FWD[cls]
        if cls == "nut":
            d_right = args_cli.nut_dock_right      # 螺母有专用右偏
        elif cls == "drill":
            d_right = args_cli.drill_dock_right    # 电钻有专用右偏
        else:
            d_right = args_cli.dock_right + BC_SPAWN_RIGHT[cls]
        return d_ahead, d_right

    def switch_to_job(k: int) -> None:
        #（--loop_all）把"所有"单目标管线重新瞄准到第 k 个作业，并复位逐次搬运的状态。调用方已经把机器人送到了A桌整备点
        #之后由常规的拾取 TURN/NAV/ALIGN 接管。
        nonlocal cur_cls, bc_policy, target_obj, dock_ahead, dock_right, \
            aim_ahead, aim_right, m3_cut, m3_zone, place_zone_xy, m3_dock_yaw, \
            m3_place_desc, place_ahead, place_right, aim_ahead_b, aim_right_b, \
            m3_route, m3_wp, m3_leg, m3_carry_hold, phase, goal_yaw, \
            retry_count, bc_last_action, bc_place_target_w, bc_obj_rest_z, \
            bc_obj_max_z, bc_report_done, bc_q_from, bc_blend_frac, \
            bc_clock, m4_job_idx
        m4_job_idx = k
        job = m4_jobs[k]
        cur_cls = job["cls"]
        bc_policy = bc_policies[cur_cls]
        target_obj = env.scene[job["entity"]]
        dock_ahead, dock_right = pick_dock_geom(cur_cls)
        aim_ahead = dock_ahead + DOCK_AHEAD_TRIM_PICK.get(cur_cls, DOCK_AHEAD_TRIM)
        aim_right = dock_right + DOCK_RIGHT_TRIM
        m3_cut = M3_CUT_TICK[cur_cls]
        m3_zone = None
        place_zone_xy = job["spot"]
        m3_dock_yaw = job["dock_yaw"]
        m3_place_desc = job["desc"]
        r_lo, r_hi, f_lo, f_hi = BC_PLACE_BOXES[cur_cls]
        place_ahead = dock_ahead + 0.5 * (f_lo + f_hi)
        place_right = dock_right + 0.5 * (r_lo + r_hi)
        aim_ahead_b = place_ahead + DOCK_AHEAD_TRIM_PLACE.get(cur_cls, DOCK_AHEAD_TRIM)
        aim_right_b = (place_right + DOCK_RIGHT_TRIM
                       + DOCK_RIGHT_TRIM_PLACE.get(cur_cls, 0.0))
        m3_route = m3_carry_route_for(place_zone_xy, m3_dock_yaw, aim_ahead_b,
                                      cur_cls == "nut")
        m3_wp = 0
        m3_leg = "pick"
        m3_carry_hold = False
        phase = "TURN"
        goal_yaw = quat_yaw(*DOCK_A_ROT)
        retry_count = 0
        # 全新的 BC 搬运状态：交接初始化会在新拾取腿的第一个 BC 步重新武装bc_clock "必须"在这里清零，
        # 不能只靠交接初始化：M3的切割检查每步都在 BC 块"之前"运行，上一个作业遗留的~930tick时钟会在新作业第一个BC步就误触发它。
        bc_clock = 0.0
        bc_last_action = torch.zeros(1, 28, device=device)
        bc_place_target_w = None
        bc_obj_rest_z = 0.0
        bc_obj_max_z = 0.0
        bc_report_done = False
        bc_q_from = None
        bc_blend_frac = 0.0
        obj_xy_j = target_obj.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]
        route_txt = " -> ".join(f"({x:.2f},{y:.2f})" for x, y in m3_route)
        print(f"[demo] ===== M4 job {k + 1}/{len(m4_jobs)}: {cur_cls} "
              f"(entity {job['entity']}) at xy=({obj_xy_j[0]:.2f},{obj_xy_j[1]:.2f}) "
              f"-> {m3_place_desc} ({place_zone_xy[0]:.2f},{place_zone_xy[1]:.2f}); "
              f"carry route {route_txt}; cut at bc t={m3_cut} =====")

    def m4_batch_summary() -> None:
        #（--repeat）统计所有已记录运行的逐作业结果百分比并打印。
        print(f"[demo] ===== M4 BATCH SUMMARY: {m4_iter} iteration(s) x "
              f"{len(m4_jobs)} job(s) =====")
        for j, jb in enumerate(m4_jobs):
            recs = [r for r in m4_records if r[0] == j]
            n = len(recs)
            if n == 0:
                continue
            cnt = {}
            for r in recs:
                key = ("PLACED-BY-PUSH" if r[3].startswith("PLACED-BY-PUSH")
                       else r[3])
                cnt[key] = cnt.get(key, 0) + 1
            parts = ", ".join(f"{k} {v}/{n}" for k, v in sorted(
                cnt.items(), key=lambda kv: -kv[1]))
            placed_err = [r[4] for r in recs if r[3] == "PLACED" and r[4] is not None]
            extra = ""
            if placed_err:
                extra = (f"; placed obj->target mean {sum(placed_err) / len(placed_err):.1f}"
                         f" / max {max(placed_err):.1f} cm")
                vecs = [(r[6], r[7]) for r in recs
                        if r[3] == "PLACED" and r[6] is not None]
                if vecs:
                    extra += (f" (vec fwd {sum(v[0] for v in vecs) / len(vecs):+.1f},"
                              f" left {sum(v[1] for v in vecs) / len(vecs):+.1f} cm)")
            retried = sum(1 for r in recs if r[5] > 0)
            print(f"[demo]   job {jb['cls']} -> {jb['desc']}: "
                  f"{100.0 * cnt.get('PLACED', 0) / n:.0f}% placed ({parts})"
                  f"{extra}; dock retried in {retried}/{n} runs")

    def m4_batch_reset() -> None:
        #（--repeat）为下一轮迭代做彻底的重新开始：环境 reset、重新加抖动、所有逐次运行的状态变量恢复到启动值、重新装载作业 1。
        #bc_gain_frac 刻意"不"动：右臂增益在第一次交接后就保持 base_fix 硬增益，与单次 loop 运行里作业 2-7 的行为完全一致。
        nonlocal obs, raise_frac, bc_frac, done_wait, m3_backoff_dist, \
            backoff_start_xy, backoff_elapsed, stand_elapsed, align_elapsed, \
            align_yaw_ref, align_stuck, retry_elapsed, gain_wait, \
            bc_settle_elapsed, m4_dropped, armret_frac, armret_from, \
            m4_results, iter_t0
        # 若上一轮死在搬运途中，握紧增益缩放可能还在生效 -> 先恢复原值
        for jid, kp1, kd1 in grip_gain_refs:
            override_joint_gains(robot, jid, kp1, kd1)
        od, _ = env.reset()
        obs = od["policy"]
        apply_spawn_jitter()
        raise_frac = 0.0
        bc_frac = 0.0
        done_wait = 0.0
        m3_backoff_dist = M3_BACKOFF_DIST
        backoff_start_xy = None
        backoff_elapsed = stand_elapsed = align_elapsed = 0.0
        align_yaw_ref, align_stuck = None, 0.0
        retry_elapsed = 0.0
        gain_wait = 0.0
        bc_settle_elapsed = 0.0
        m4_dropped = False
        armret_frac = 0.0
        armret_from = None
        m4_results = []
        switch_to_job(0)
        iter_t0 = t

    if args_cli.loop_all:
        switch_to_job(0)                # M4：立即瞄准第一个作业

    # ---- --record：为每台调试相机开一个带时间戳的视频输出 ----
    rec_sinks = None
    rec_acc = 0.0                       # 仿真时间累加器：保证精确 30 Hz 采样
    if args_cli.record:
        rec_label = {"cam_table_a": "tableA", "cam_table_b": "tableB"}
        cams = [n for n in rec_label if n in env.scene.sensors]
        if cams:
            os.makedirs(args_cli.video_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            rec_sinks = {n: _VideoSink(os.path.join(args_cli.video_dir,
                                                    f"{stamp}_{rec_label[n]}"), REC_FPS)
                         for n in cams}
            print(f"[demo] recording {len(cams)} debug cams at {REC_FPS} fps, "
                  f"1x real-time -> {args_cli.video_dir}/{stamp}_tableA/_tableB")
        else:
            print("[demo] --record: no debug cameras in this scene (flat A/B variant?) - recording off")

    # ==================================================================
    # 主循环：每次迭代 = 一个 50 Hz 控制步
    # ==================================================================
    while simulation_app.is_running() and t < args_cli.max_time:
        # ---- --repeat：逐迭代门控 ----
        # 一次死锁的运行（无止境的 RETRY 乒乓、安全网漏掉的原地打转……）
        # 不能吃掉整个通宵批量：在 作业数 x --iter_cap 处强制切断本迭代，把没跑完的作业统统判 TIMEOUT。
        if (args_cli.loop_all and args_cli.repeat > 1 and not m4_iter_end
                and t - iter_t0 > m4_iter_cap):
            print(f"[demo] --repeat: iteration {m4_iter} hit the {m4_iter_cap:.0f} s cap "
                  f"stuck in {phase} (job {len(m4_results) + 1}/{len(m4_jobs)}) - "
                  f"scoring the unfinished jobs TIMEOUT")
            for j in range(len(m4_results), len(m4_jobs)):
                jb = m4_jobs[j]
                m4_records.append((j, jb["cls"], jb["desc"], "TIMEOUT", None,
                                   retry_count if j == m4_job_idx else 0,
                                   None, None))
                m4_results.append(f"job {j + 1}/{len(m4_jobs)} {jb['cls']} -> "
                                  f"{jb['desc']}: TIMEOUT (stuck in {phase})")
            m4_iter_end = True

        # ---- 本步状态快照（python 浮点数；单环境） ----
        base_xy = robot.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]  # 骨盆平面位置
        qw, qx, qy, qz = robot.data.root_quat_w[0].tolist()
        yaw = quat_yaw(qw, qx, qy, qz)      # 实时偏航角
        # 偏航补偿的动态停靠目标
        if m3_leg == "return":
            # M4返回腿（B 桌 -> A 桌）：目标是当前返回路线航点；BACKOFF
            # 保持停靠朝向，TURN/TRANSIT 朝航点转向。最后一个航点是A桌整备点——到达即切换到下一个作业的拾取。
            goal_xy = m3_route[min(m3_wp, len(m3_route) - 1)]
        elif m3_leg in ("carry", "place"):
            # M3 搬运/放置腿。走"路线"期间（BACKOFF/TURN/TRANSIT且还剩航点）目标是当前航点；
            #从 FACE 开始改为围绕（静态）放置点的偏航补偿动态停靠，与 A 桌完全一样，只是换成放置目标的瞄准偏移。
            if phase in ("BACKOFF", "TURN", "TRANSIT") and m3_wp < len(m3_route):
                goal_xy = m3_route[m3_wp]
            else:
                # ALIGN 及之后用实时偏航（偏航补偿），接近阶段用名义停靠朝向
                yaw_ref = yaw if phase in ("ALIGN", "STAND", "GAIN", "DONE", "BC") else goal_yaw
                yc, ys = math.cos(yaw_ref), math.sin(yaw_ref)   # 前向=(yc,ys)，右向=(ys,-yc)
                # 骨盆目标 = 放置点往后退 aim_ahead_b、往左挪 aim_right_b
                goal_xy = (place_zone_xy[0] - aim_ahead_b * yc - aim_right_b * ys,
                           place_zone_xy[1] - aim_ahead_b * ys + aim_right_b * yc)
        elif target_obj is not None:
            # 拾取腿的动态 dock：目标 = 工件"实时"位置反推出的骨盆停靠点
            yaw_ref = yaw if phase in ("ALIGN", "STAND", "GAIN", "DONE", "BC") else goal_yaw
            yc, ys = math.cos(yaw_ref), math.sin(yaw_ref)   # 前向=(yc,ys)，右向=(ys,-yc)
            obj_xy_l = target_obj.data.root_pos_w[0, :2] - env.scene.env_origins[0, :2]
            goal_xy = (obj_xy_l[0].item() - aim_ahead * yc - aim_right * ys,
                       obj_xy_l[1].item() - aim_ahead * ys + aim_right * yc)
        ex, ey = goal_xy[0] - base_xy[0].item(), goal_xy[1] - base_xy[1].item()
        dist = math.hypot(ex, ey)                         # 到目标的平面距离
        head_err = wrap_angle(math.atan2(ey, ex) - yaw)   # "面朝目标"的朝向误差
        goal_yaw_err = wrap_angle(goal_yaw - yaw)         # 与停靠朝向的偏航误差
        # 位置误差旋进机体系（供 ALIGN 伺服使用）：ex_b 前向、ey_b 左向
        ex_b = math.cos(yaw) * ex + math.sin(yaw) * ey
        ey_b = -math.sin(yaw) * ex + math.cos(yaw) * ey

        # ---- M4 掉件监视 ----
        # 搬运途中掉件会让整条放置被放弃
        if (args_cli.loop_all and m3_leg == "carry" and phase != "ARMRET"
                and target_obj is not None):
            # 掉件判据两条：工件离手腕太远（脱手）或工件高度掉到地板附近
            drop_grip = torch.norm(target_obj.data.root_pos_w[0]
                                   - robot.data.body_pos_w[0, rw_id]).item()
            drop_obj_z = target_obj.data.root_pos_w[0, 2].item()
            if drop_grip > M4_DROP_GRIP or drop_obj_z < M4_DROP_OBJ_Z:
                m4_records.append((len(m4_results), cur_cls, m3_place_desc,
                                   "DROPPED", None, retry_count, None, None))
                m4_results.append(
                    f"job {m4_job_idx + 1}/{len(m4_jobs)} {cur_cls} -> "
                    f"{m3_place_desc}: DROPPED during carry "
                    f"(grip {drop_grip:.2f} m, obj_z {drop_obj_z:.2f} m)")
                if args_cli.repeat > 1 and len(m4_results) >= len(m4_jobs):
                    m4_iter_end = True      # 已是本迭代最后一个作业
                print(f"[demo] t={t:5.1f}s JOB {m4_job_idx + 1} DROPPED the "
                      f"{cur_cls} mid-carry (grip {drop_grip:.2f} m, obj_z "
                      f"{drop_obj_z:.2f} m) - skipping the place leg, raising "
                      f"the arm and moving on")
                phase = "ARMRET"
                armret_frac = 0.0
                armret_from = robot.data.joint_pos[:, arm_ids].clone()
                m3_carry_hold = True    # ARMRET 期间手臂目标归它管
                m4_dropped = True       # 掉在半路：ARMRET 出口跳过相对停靠点的后退几何
                # 关闭搬运
                for jid, kp1, kd1 in grip_gain_refs:
                    override_joint_gains(robot, jid, kp1, kd1)

        # ==================================================================
        # 航点状态机 -> 速度指令 (vx 前向, vy 左向, wz 偏航角速度)
        # ==================================================================
        vx = vy = wz = 0.0
        # 螺母搬运温柔化：螺母骑在最长的力臂末端；持物期间把偏航/侧移
        # 指令包络减半
        if cur_cls == "nut" and m3_leg == "carry":
            wz_cap, vy_cap = M3_CARRY_WZ_CAP_NUT, M3_CARRY_VY_CAP_NUT
        else:
            wz_cap, vy_cap = M3_CARRY_WZ_CAP, None
        if phase == "BACKOFF":
            # M3：抓取冻结着从A桌正后方退开，使接下来 ~180 度转身时
            # 手臂+工件的扫掠范围能躲开桌子。不给偏航指令——后退期间保持停靠朝向。
            backoff_elapsed += CTRL_DT
            vx = RETRY_VX                       # 负值 = 倒退
            back = math.hypot(base_xy[0].item() - backoff_start_xy[0],
                              base_xy[1].item() - backoff_start_xy[1])
            # 退够距离或超时均可退出（防止卡死在后退里）
            if back >= m3_backoff_dist or backoff_elapsed >= M3_BACKOFF_MAX_SEC:
                phase = "TURN"
                if m3_leg == "return":
                    nxt_txt = ("home" if m4_job_idx + 1 >= len(m4_jobs)
                               else f"job {m4_job_idx + 2}")
                    print(f"[demo] t={t:5.1f}s backed off {back:.2f} m from table B - "
                          f"turning right toward table A (next: {nxt_txt})")
                else:
                    print(f"[demo] t={t:5.1f}s backed off {back:.2f} m from table A - "
                          f"turning toward the table-B dock ({m3_place_desc})")
        elif phase == "TURN":
            # -------- TURN：原地转身面向目标 --------
            if (m3_leg in ("carry", "return")
                    and m3_wp < len(m3_route) - 1
                    and dist < M3_WP_RADIUS):
                # 安全网。直接消费掉该航点、瞄准下一个（dist/head_err 下一拍自动刷新）。
                m3_wp += 1
                print(f"[demo] t={t:5.1f}s TURN: waypoint {m3_wp}/{len(m3_route)} "
                      f"already underfoot - skipping to "
                      f"({m3_route[m3_wp][0]:.2f},{m3_route[m3_wp][1]:.2f})")
            wz = 1.5 * head_err                 # P 控制：朝向误差 x 1.5 增益
            if m3_leg in ("carry", "return"):
                # 指定方向：离开两侧停靠点时都"向右"转（顺时针）。误差接近180度时符号本来就是任意的，在误差还很大时强制方向。
                if abs(head_err) > 1.8:
                    wz = -abs(wz)
                wz = max(-wz_cap, min(wz_cap, wz))  # 持物期限幅
            if abs(head_err) < TURN_FACE_TOL:
                # 已面向目标：有剩余航点走 TRANSIT，否则直接 NAV
                phase = ("TRANSIT" if (m3_leg in ("carry", "return")
                                       and m3_wp < len(m3_route)) else "NAV")
        elif phase == "TRANSIT":
            # -------- TRANSIT：M3 航点路线巡航 --------
            # 朝当前航点做 NAV 式转向；最后一个航点是停靠轴上离桌 1 m 的"整备点"。
            last_wp = m3_wp == len(m3_route) - 1
            if abs(head_err) > M3_TRANSIT_TURN_ERR:
                # 误差过大：只转不走，且转速下限保证步态真的响应
                wz = math.copysign(max(abs(1.5 * head_err), ALIGN_CMD_FLOOR_WZ), head_err)
                wz = max(-wz_cap, min(wz_cap, wz))
            else:
                # 最后一个航点：0.8*dist的刹车斜坡
                vx = min((0.8 if last_wp else 1.5) * dist, M3_CARRY_VX_CAP)
                vy = 0.5 * ey_b
                if vy_cap is not None:
                    vy = max(-vy_cap, min(vy_cap, vy))
                wz = max(-wz_cap, min(wz_cap, 1.5 * head_err))
            if dist < (M3_WP_RADIUS_LAST if last_wp else M3_WP_RADIUS):
                m3_wp += 1                      # 到达当前航点，推进到下一个
                if m3_wp >= len(m3_route):
                    if m3_leg == "return":
                        if m4_job_idx + 1 >= len(m4_jobs):
                            # M4：最后一个作业后回到出生点并汇报
                            m4_done = True
                            print(f"[demo] t={t:5.1f}s back at the start point "
                                  f"({base_xy[0].item():.2f},{base_xy[1].item():.2f})")
                            print(f"[demo] ===== M4 LOOP COMPLETE ({len(m4_jobs)} jobs) =====")
                            for line in m4_results:
                                print(f"[demo]   {line}")
                            break
                        # M4：回到 A 桌整备点（离 A 桌 2.5 m、正对下一件
                        # 工件所在行）——启动下一个作业
                        print(f"[demo] t={t:5.1f}s re-staged 2.5 m in front of "
                              f"table A - turning onto the next pick dock")
                        switch_to_job(m4_job_idx + 1)
                    else:
                        phase = "FACE"
                        print(f"[demo] t={t:5.1f}s staged {M3_STAGE_DIST:.1f} m off the "
                              f"table-B dock - turning onto the dock heading "
                              f"({math.degrees(m3_dock_yaw):.0f} deg)")
                else:
                    print(f"[demo] t={t:5.1f}s route waypoint {m3_wp}/{len(m3_route)} "
                          f"reached - next ({m3_route[m3_wp][0]:.2f},{m3_route[m3_wp][1]:.2f})")
        elif phase == "FACE":
            # -------- FACE：原地转到停靠朝向 --------
            # 在最后 1 m 正面接近"之前"就原地转上停靠朝向，让 ALIGN 一开始就几乎对齐，而不是贴着桌子横着蟹行
            wz = 1.5 * goal_yaw_err
            if abs(goal_yaw_err) > 0.05:
                # 转速下限：太小的 wz 步态不响应
                wz = math.copysign(max(abs(wz), ALIGN_CMD_FLOOR_WZ), goal_yaw_err)
            wz = max(-wz_cap, min(wz_cap, wz))
            # 出口容差 0.12 -> 0.06 rad
            if abs(goal_yaw_err) < 0.06:
                phase = "NAV"
                print(f"[demo] t={t:5.1f}s facing the dock (yaw_err "
                      f"{math.degrees(goal_yaw_err):+.1f} deg) - final head-on approach")
        elif phase == "NAV":
            # -------- NAV：直线接近目标 --------
            vx = min(1.5 * dist, M3_CARRY_VX_CAP if m3_leg == "carry" else CMD_LIM_VX[1])
            vy = 0.5 * ey_b                     # 侧向微调
            wz = 1.5 * head_err                 # 持续修正朝向
            if m3_leg == "carry":
                # 持物期收紧侧移/转速包络
                if vy_cap is not None:
                    vy = max(-vy_cap, min(vy_cap, vy))
                wz = max(-wz_cap, min(wz_cap, wz))
            if dist < NAV_DONE_DIST:
                phase = "ALIGN"
                align_elapsed = 0.0             # 每次进入都重置升级计时
                align_yaw_ref, align_stuck = None, 0.0
        elif phase == "ALIGN":
            # -------- ALIGN：停靠点精对齐 --------
            # 机体系下的慢速位置伺服 + 转到停靠朝向
            align_elapsed += CTRL_DT
            vx = 0.8 * ex_b                     # 前向 P 伺服
            vy = 0.8 * ey_b                     # 侧向 P 伺服
            if abs(ex_b) > ALIGN_FLOOR_ENGAGE:
                
                vx_floor = min(ALIGN_CMD_FLOOR_X_MAX,
                               ALIGN_CMD_FLOOR + ALIGN_X_ESCALATE * int(align_elapsed / 2.0))
                vx = math.copysign(max(abs(vx), vx_floor), ex_b)
            if abs(ey_b) > ALIGN_FLOOR_ENGAGE:
                
                vy_floor = min(CMD_LIM_VY[1],
                               ALIGN_CMD_FLOOR_Y + ALIGN_Y_ESCALATE * int(align_elapsed / 2.0))
                vy = math.copysign(max(abs(vy), vy_floor), ey_b)
            wz = 1.5 * goal_yaw_err
            if abs(goal_yaw_err) > 0.02:
                
                wz_floor = min(ALIGN_CMD_FLOOR_WZ_MAX,
                               ALIGN_CMD_FLOOR_WZ + ALIGN_WZ_ESCALATE * int(align_elapsed / 2.0))
                wz = math.copysign(max(abs(wz), wz_floor), goal_yaw_err)
            
            if align_yaw_ref is None or abs(goal_yaw_err - align_yaw_ref) > ALIGN_STUCK_EPS:
                align_yaw_ref, align_stuck = goal_yaw_err, 0.0  # 有变化 -> 重新计时
            else:
                align_stuck += CTRL_DT                          # 没变化 -> 累计卡住时长
            if dist < ALIGN_DONE_DIST and abs(goal_yaw_err) < ALIGN_DONE_YAW:
                # 正常出口：位置、偏航双双达标 -> 进 STAND
                phase = "STAND"
                stand_elapsed = 0.0
                stand_start_xy = (base_xy[0].item(), base_xy[1].item())
                print(f"[demo] t={t:5.1f}s docked - holding zero command for {args_cli.stand_time:.0f}s")
            elif align_stuck >= ALIGN_STUCK_SEC or align_elapsed >= ALIGN_MAX_SEC:
                # 死锁或超时：分三种处置
                if dist < ALIGN_DONE_DIST and abs(goal_yaw_err) <= PASS_YAW:
                    # 静态平衡 + 位置良好 + 残余偏航无害 -> ACCEPTED 放行
                    phase = "STAND"
                    stand_elapsed = 0.0
                    stand_start_xy = (base_xy[0].item(), base_xy[1].item())
                    print(f"[demo] t={t:5.1f}s docked (ACCEPTED with residual yaw "
                          f"{math.degrees(goal_yaw_err):+.1f} deg after {align_stuck:.0f}s deadlock) - "
                          f"holding zero command for {args_cli.stand_time:.0f}s")
                elif retry_count < DOCK_RETRY_MAX:
                    # 又卡住又超差：后退 + 重新接近
                    phase = "RETRY"
                    retry_elapsed = 0.0
                    retry_count += 1
                    print(f"[demo] t={t:5.1f}s ALIGN deadlocked at dist={dist:.2f} m "
                          f"yaw_err={math.degrees(goal_yaw_err):+.1f} deg - "
                          f"RETRY #{retry_count}: backing off to re-approach")
                else:
                    # 重试次数耗尽：带着现状继续，报告（STAND 会打分；M1 报告会显示残差）
                    phase = "STAND"
                    stand_elapsed = 0.0
                    stand_start_xy = (base_xy[0].item(), base_xy[1].item())
                    print(f"[demo] t={t:5.1f}s ALIGN deadlocked with retries exhausted "
                          f"({retry_count}) - docking AS IS at dist={dist:.2f} m "
                          f"yaw_err={math.degrees(goal_yaw_err):+.1f} deg")
        elif phase == "RETRY":
            # -------- RETRY：后退再来 --------
            # 从桌边退开；机器人一边移动一边迈步，偏航指令重新起作用——修正朝向
            retry_elapsed += CTRL_DT
            vx = RETRY_VX                       # 倒退
            wz = 1.5 * goal_yaw_err             # 边退边修偏航
            if dist > RETRY_BACKOFF_DIST or retry_elapsed > RETRY_MAX_SEC:
                phase = "NAV"                   # NAV -> ALIGN 会重置对齐状态
                print(f"[demo] t={t:5.1f}s RETRY #{retry_count} backed off to "
                      f"dist={dist:.2f} m yaw_err={math.degrees(goal_yaw_err):+.1f} deg - re-approaching")
        elif phase == "STAND":
            # -------- STAND：零指令静立打分 --------
            stand_elapsed += CTRL_DT
            if stand_elapsed >= args_cli.stand_time:
                # 静立漂移：站立期间骨盆挪动了多少
                drift = math.hypot(base_xy[0].item() - stand_start_xy[0],
                                   base_xy[1].item() - stand_start_xy[1])
                if m3_leg == "carry":
                    # M3 B 桌门限：把"放置区"在机器人系下的位姿与训练时的
                    # 放置目标偏移逐轴对比打分——续跑的放置将继承的几何。
                    zdx = place_zone_xy[0] - base_xy[0].item()
                    zdy = place_zone_xy[1] - base_xy[1].item()
                    # 旋进机体系：zone_fwd 前向 / zone_lat 左向
                    zone_fwd = math.cos(yaw) * zdx + math.sin(yaw) * zdy
                    zone_lat = -math.sin(yaw) * zdx + math.cos(yaw) * zdy
                    fwd_err = zone_fwd - place_ahead       # 正 = 放置区太深
                    lat_err = zone_lat + place_right       # 正 = 放置区偏左太多
                    grip = torch.norm(target_obj.data.root_pos_w[0]
                                      - robot.data.body_pos_w[0, rw_id]).item()
                    fwd_gate_b = PASS_AXIS_FWD_PLACE.get(cur_cls or "", PASS_AXIS)
                    ok = (abs(fwd_err) <= fwd_gate_b and abs(lat_err) <= PASS_AXIS
                          and abs(goal_yaw_err) <= PASS_YAW)
                    print(
                        f"[demo] ===== M1-B REPORT ({'PASS' if ok else 'FAIL'}) =====\n"
                        f"[demo]   zone fwd err : {fwd_err * 100:+5.1f} cm (gate +-{fwd_gate_b * 100:.1f}, + = deep)\n"
                        f"[demo]   zone lat err : {lat_err * 100:+5.1f} cm (gate +-{PASS_AXIS * 100:.1f}, + = left)\n"
                        f"[demo]   dock yaw err : {math.degrees(goal_yaw_err):5.1f} deg (gate {math.degrees(PASS_YAW):.1f} deg)\n"
                        f"[demo]   stand drift  : {drift * 100:5.1f} cm over {args_cli.stand_time:.0f}s\n"
                        f"[demo]   pelvis height: {robot.data.root_pos_w[0, 2].item():.3f} m\n"
                        f"[demo]   grip (hand->obj): {grip:.3f} m, obj_z {target_obj.data.root_pos_w[0, 2].item():.3f} m")
                elif target_obj is not None:
                    # 把"工件"实际的机器人系位姿与"训练时"的停靠偏移"逐轴"打分——这就是 BC 将继承的几何。
                    odx = target_obj.data.root_pos_w[0, 0].item() - robot.data.root_pos_w[0, 0].item()
                    ody = target_obj.data.root_pos_w[0, 1].item() - robot.data.root_pos_w[0, 1].item()
                    # 工件相对骨盆的向量旋进机体系
                    piece_fwd = math.cos(yaw) * odx + math.sin(yaw) * ody
                    piece_lat = -math.sin(yaw) * odx + math.cos(yaw) * ody
                    fwd_err = piece_fwd - dock_ahead       # 正 = 工件太深
                    lat_err = piece_lat + dock_right       # 正 = 工件偏左太多
                    fwd_gate = PASS_AXIS_FWD_PICK.get(cur_cls or "", PASS_AXIS)
                    # 类别专属的非对称通过窗口（没有就用对称默认值）
                    win = PASS_WINDOW_PICK.get(cur_cls or "", {})
                    f_lo, f_hi = win.get("fwd", (-fwd_gate, fwd_gate))
                    l_lo, l_hi = win.get("lat", (-PASS_AXIS, PASS_AXIS))
                    y_lo, y_hi = win.get("yaw", (-PASS_YAW, PASS_YAW))
                    ok = (f_lo <= fwd_err <= f_hi and l_lo <= lat_err <= l_hi
                          and y_lo <= goal_yaw_err <= y_hi)
                    print(
                        f"[demo] ===== M1 REPORT ({'PASS' if ok else 'FAIL'}) =====\n"
                        f"[demo]   piece fwd err: {fwd_err * 100:+5.1f} cm (gate {f_lo * 100:+.1f}..{f_hi * 100:+.1f}, + = deep)\n"
                        f"[demo]   piece lat err: {lat_err * 100:+5.1f} cm (gate {l_lo * 100:+.1f}..{l_hi * 100:+.1f}, + = left)\n"
                        f"[demo]   dock yaw err : {math.degrees(goal_yaw_err):5.1f} deg (gate {math.degrees(y_lo):+.1f}..{math.degrees(y_hi):+.1f} deg)\n"
                        f"[demo]   stand drift  : {drift * 100:5.1f} cm over {args_cli.stand_time:.0f}s\n"
                        f"[demo]   pelvis height: {robot.data.root_pos_w[0, 2].item():.3f} m")
                else:
                    # 无目标工件（平地模式等）：只按距离+偏航打分
                    ok = dist <= PASS_DIST and abs(goal_yaw_err) <= PASS_YAW
                    print(
                        f"[demo] ===== M1 REPORT ({'PASS' if ok else 'FAIL'}) =====\n"
                        f"[demo]   dock pos err : {dist * 100:5.1f} cm (gate {PASS_DIST * 100:.0f} cm)\n"
                        f"[demo]   dock yaw err : {math.degrees(goal_yaw_err):5.1f} deg (gate {math.degrees(PASS_YAW):.1f} deg)\n"
                        f"[demo]   stand drift  : {drift * 100:5.1f} cm over {args_cli.stand_time:.0f}s\n"
                        f"[demo]   pelvis height: {robot.data.root_pos_w[0, 2].item():.3f} m")
                if not ok and retry_count < DOCK_RETRY_MAX:
                    # dist 是对"偏航补偿后目标"量的，也就是说它"就是"BC 将
                    # 继承的工件机器人系摆放误差——重新停靠
                    phase = "RETRY"
                    retry_elapsed = 0.0
                    retry_count += 1
                    print(f"[demo] t={t:5.1f}s M1 gate failed - "
                          f"RETRY #{retry_count}: backing off to re-dock")
                else:
                    if not ok:
                        print(f"[demo] WARNING: M1 gate still failing after "
                              f"{retry_count} retries - proceeding anyway")
                    if m3_leg == "carry":
                        # M3：抓取仍冻结着停靠在 B 桌。base_fix增益已经生效——直接续跑BC时钟，不走GAINramp（重启ramp会先跌回软增益）。
                        phase = "BC"
                        print(f"[demo] t={t:5.1f}s docked at table B ({m3_place_desc}) - "
                              f"resuming the BC place leg at tick {m3_cut} NOW "
                              f"(gains already base_fix, grasp targets frozen throughout)")
                    elif args_cli.bc_expert:
                        # 抬臂入场：跳过 DONE 的低位下放。
                        # 增益 ramp期间保持行走抬臂位，BC的第一个动作是抬臂 -> 抓取点上方悬停。
                        phase = "GAIN"
                        gain_wait = 0.0
                        print(f"[demo] t={t:5.1f}s M1 scored - RAISED-START expert: "
                              f"holding walk-raise pose while base_fix gains ramp "
                              f"({GAIN_RAMP_TIME:.1f}s), then BC (no low descent)")
                    elif bc_policy is not None and args_cli.bc_raised:
                        # --bc_raised：网络走与采集专家"相同的"GAIN 通道入场，使 bc t=0 时的手臂状态
                        # 等于所录数据集的 t=0（行走抬臂位），而不是旧的固定底座起始位。
                        phase = "GAIN"
                        gain_wait = 0.0
                        print(f"[demo] t={t:5.1f}s M1 scored - [--bc_raised] NET will take over at "
                              f"the RAISED walk pose (matches re-collected dataset t=0; "
                              f"no DONE low descent); gains ramp {GAIN_RAMP_TIME:.1f}s first")
                    elif bc_policy is not None and args_cli.bc_immediate:
                        # 诊断通道：从停靠点起 BC 直接接管手臂；
                        # 其目标从抬臂位混合切入
                        # （--bc_blend）。下面的 BC 块会在它的第一步执行交接初始化。
                        phase = "BC"
                        print(f"[demo] t={t:5.1f}s M1 scored - IMMEDIATE BC takeover, "
                              f"blending from the raised arm pose over {args_cli.bc_blend:.1f}s")
                    else:
                        phase = "DONE"          # 默认分阶段路径：先下放到 BC 起始位
                        done_wait = 0.0
                    if (bc_active and bc_gain_plan and m3_leg == "pick"
                            and bc_gain_frac is None):
                        # 把右臂+双手朝base_fix硬化（下放和 BC 都需要）是 RAMP 渐变、不是硬切：见 GAIN_RAMP_TIME。
                        bc_gain_frac = 0.0
                        print(f"[demo] ramping base_fix PD gains onto "
                              f"{len(bc_gain_plan)} right-arm/hand joints over "
                              f"{GAIN_RAMP_TIME:.1f}s")
        elif phase == "GAIN":
            # -------- GAIN：抬臂入场的增益等待段 --------
            # 保持零底座指令 + 抬臂位；等 base_fix 增益 ramp 完成后再交给专家/网络。
            gain_wait += CTRL_DT
            if (bc_gain_frac is None or bc_gain_frac >= 1.0) and gain_wait >= GAIN_RAMP_TIME:
                phase = "BC"
                who = ("NET driving + expert labelling (--bc_dagger)" if args_cli.bc_dagger
                       else "expert" if expert is not None else "NET (--bc_raised)")
                print(f"[demo] t={t:5.1f}s base_fix gains ready - RAISED-START BC "
                      f"takes over: {who} now owns the right arm at the walk-raise "
                      f"pose (no DONE descent)")
        elif phase == "DONE":
            # -------- DONE：站立 + 手臂下放到 BC 起始位 --------
            # 默认分阶段路径：保持站立（零指令）。要求：手必须完全"到位"之后 BC才能接管——下面的门限用"实测"右臂关
            # 节对照 BC 起始位（位置"和"速度都要过），且到位状态必须持续--bc_settle 秒；任何一次破坏都会重置计时
            # 绝不在运动中交接。交接初始化本体在 BC 块里（与 --bc_immediate 共用）。
            if bc_active and bc_frac >= 1.0:
                done_wait += CTRL_DT
                # 实测到位检查：所有右臂关节的最大位置误差与最大速度
                arr_err = (robot.data.joint_pos[0, arm_ids] - q_arm_bc[0]).abs().max().item()
                arr_vel = robot.data.joint_vel[0, arm_ids].abs().max().item()
                if arr_err <= BC_ARRIVE_POS_TOL and arr_vel <= BC_ARRIVE_VEL_TOL:
                    bc_settle_elapsed += CTRL_DT
                else:
                    bc_settle_elapsed = 0.0     # 未到位 / 仍在运动 -> 重新计时
                if bc_settle_elapsed >= args_cli.bc_settle:
                    phase = "BC"
                    print(f"[demo] t={t:5.1f}s right arm ARRIVED at the BC start pose "
                          f"(max joint err {arr_err:.3f} rad, max vel {arr_vel:.2f} rad/s, "
                          f"held {args_cli.bc_settle:.1f}s) - BC takes over NOW")
                elif done_wait >= BC_ARRIVE_TIMEOUT:
                    resid = " ".join(
                        f"{n}={robot.data.joint_pos[0, j].item() - q_arm_bc[0, k].item():+.3f}"
                        for k, (j, n) in enumerate(zip(arm_ids, arm_names)))
                    print(f"[demo] WARNING: t={t:5.1f}s arm NOT arrived after "
                          f"{BC_ARRIVE_TIMEOUT:.0f}s (err {arr_err:.3f} rad, vel {arr_vel:.2f} "
                          f"rad/s) - handing over anyway. residuals: {resid}")
                    phase = "BC"
        elif phase == "BC":
            pass  # 零底座指令；BC 策略在下方驱动上半身
        elif phase == "ARMRET":
            # -------- ARMRET：放置结束后的手臂回收 --------
            # M4：放置腿已出分——零底座指令，右臂从放置腿的结束姿态 ramp回"行走抬臂位"ramp 本体写在下方的手臂目标段。
            armret_frac = min(1.0, armret_frac + CTRL_DT / M4_ARM_RETURN_SEC)
            if armret_frac >= 1.0:
                m3_carry_hold = False   # 默认的保持写入维持抬臂位
                m3_leg = "return"
                m3_wp = 0
                # 返回路线新方案：BACKOFF 自己就把撤离线走完——一直直退到骨盆到达走廊
                # （区位停靠：向北退到 y=M4_B_EXIT_Y；螺母停靠：向东退到 x=M4_B_EXIT_X_EAST），然后"一次"右转上走廊向西走。
                if m4_job_idx + 1 < len(m4_jobs):
                    # 目的地：正对"下一个"拾取停靠点的 A 桌整备点
                    nxt = m4_jobs[m4_job_idx + 1]
                    n_obj = env.scene[nxt["entity"]]
                    _, n_right = pick_dock_geom(nxt["cls"])
                    # 下一个停靠点的 y = 工件 y - (右偏 + 瞄准修正)
                    n_dock_y = (n_obj.data.root_pos_w[0, 1].item()
                                - env.scene.env_origins[0, 1].item()
                                - (n_right + DOCK_RIGHT_TRIM))
                    dest = (M4_A_RESTAGE_X, n_dock_y)
                else:
                    # 最后一个作业已完成：沿同一撤离走廊走回出生点，然后结束。
                    dest = (ROBOT_START_POS[0], ROBOT_START_POS[1])
                if m4_dropped:
                    # 半路掉件：机器人"不在"B 桌停靠点，下面那套相对停靠点的撤离几何不适用。
                    # 标准短后退，然后直奔目的地
                    # 若仍在走廊出口以东就先经过西侧走廊出口（与 B 桌保持 >=1 m），否则直走。
                    m4_dropped = False
                    m3_backoff_dist = M3_BACKOFF_DIST
                    m3_route = []
                    if base_xy[0].item() > M4_B_EXIT_X_WEST:
                        m3_route.append((M4_B_EXIT_X_WEST, M4_B_EXIT_Y))
                    m3_route.append(dest)
                elif cur_cls == "nut":
                    # 面朝 -x：后退等于往 +x（向东）移动
                    m3_backoff_dist = max(M3_BACKOFF_DIST,
                                          M4_B_EXIT_X_EAST - base_xy[0].item())
                    m3_route = [(M4_B_EXIT_X_EAST, M4_B_EXIT_Y),
                                (M4_B_EXIT_X_WEST, M4_B_EXIT_Y),
                                dest]
                else:
                    # 面朝 -y：后退等于往 +y（向北，离开桌子）移动
                    m3_backoff_dist = max(M3_BACKOFF_DIST,
                                          M4_B_EXIT_Y - base_xy[1].item())
                    m3_route = [(M4_B_EXIT_X_WEST, M4_B_EXIT_Y),
                                dest]
                goal_yaw = quat_yaw(*DOCK_A_ROT)
                phase = "BACKOFF"
                backoff_start_xy = (base_xy[0].item(), base_xy[1].item())
                backoff_elapsed = 0.0
                route_txt = " -> ".join(f"({x:.2f},{y:.2f})" for x, y in m3_route)
                print(f"[demo] t={t:5.1f}s arm back at the walk-raise pose - "
                      f"backing {m3_backoff_dist:.2f} m onto the exit corridor, "
                      f"then ONE right turn; return route {route_txt}")

        # 把速度指令写进训练观测项读取的"同一个"缓冲区（限幅后）
        # ——这就是"把指令喂给行走 PPO 策略"的全部机制
        cmd_term.vel_command_b[0, 0] = max(CMD_LIM_VX[0], min(CMD_LIM_VX[1], vx))
        cmd_term.vel_command_b[0, 1] = max(CMD_LIM_VY[0], min(CMD_LIM_VY[1], vy))
        cmd_term.vel_command_b[0, 2] = max(CMD_LIM_WZ[0], min(CMD_LIM_WZ[1], wz))

        # ---- 右臂：行走期抬起，进入 DONE 后 ramp 到 BC 起始位
        # （动作管理器只写它的 19 个关节；这里写的目标会一直生效） ----
        raise_frac = min(1.0, raise_frac + raise_step)   # 抬臂进度 0->1
        if phase == "DONE":
            bc_frac = min(1.0, bc_frac + bc_step)        # 下放进度 0->1
        # 两段插值：默认位 -> 抬臂位 -> BC 起始位
        q_walk = q_arm_default + raise_frac * (q_arm_walk - q_arm_default)
        q_arm_target = q_walk + bc_frac * (q_arm_bc - q_walk)
        # BC期间这些关节归策略管；M3 搬运腿期间"冻结的"抓取目标必须在行走中存活——在这里覆盖那
        # 7 个手臂关节会把手臂 ramp 回抬臂位、把工件掉在地上。
        if phase == "ARMRET" and armret_from is not None:
            # M4 放置后手臂回收：7 个手臂关节用平滑插值从捕获的放置腿结束姿态回到抬臂行走位
            q_ret = armret_from + _ease(armret_frac) * (q_arm_walk - armret_from)
            robot.set_joint_position_target(q_ret, joint_ids=arm_ids)
        elif phase != "BC" and not m3_carry_hold:
            robot.set_joint_position_target(q_arm_target, joint_ids=arm_ids)

        # ---- base_fix PD 增益 ramp（STAND 交接时启动）----
        if bc_active and bc_gain_frac is not None and bc_gain_frac < 1.0:
            bc_gain_frac = min(1.0, bc_gain_frac + CTRL_DT / GAIN_RAMP_TIME)
            for jid, kp0, kd0, kp1, kd1 in bc_gain_plan:
                # 线性插值 kp/kd：soft(kp0,kd0) -> base_fix(kp1,kd1)
                override_joint_gains(robot, jid,
                                     kp0 + bc_gain_frac * (kp1 - kp0),
                                     kd0 + bc_gain_frac * (kd1 - kd0))
            if bc_gain_frac >= 1.0:
                print(f"[demo] base_fix PD gains fully applied to "
                      f"{len(bc_gain_plan)} right-arm/hand joints")

        # ---- M3 切割（--carry_to_b）：拾取腿跑到HOLD2结束——冻结抓取然后步行去B桌停靠点。BC时钟停在切割tick，等重新进入时续跑。
        if (phase == "BC" and args_cli.carry_to_b and m3_leg == "pick"
                and bc_clock >= m3_cut):
            m3_leg = "carry"
            m3_carry_hold = True        # 抓取目标冻结，禁止被覆盖
            phase = "BACKOFF"
            backoff_start_xy = (base_xy[0].item(), base_xy[1].item())
            backoff_elapsed = 0.0
            m3_backoff_dist = M3_BACKOFF_DIST
            goal_yaw = m3_dock_yaw
            m3_wp = 0                   # 从头走整条整备路线
            retry_count = 0             # B 桌停靠享有全新的重试预算
            # 搬运握紧：为行走加硬右手手指增益（目标保持冻结不变；
            for jid, kp1, kd1 in grip_gain_refs:
                override_joint_gains(robot, jid, kp1 * M3_CARRY_GRIP_KP_SCALE, kd1)
            if grip_gain_refs:
                print(f"[demo] carry clench: right-hand finger kp x"
                      f"{M3_CARRY_GRIP_KP_SCALE:.1f} on {len(grip_gain_refs)} "
                      f"joints until the place handover")
            # 切割时刻的诊断量：手腕-工件距离 + 到目前为止的峰值抬升
            grip = torch.norm(target_obj.data.root_pos_w[0]
                              - robot.data.body_pos_w[0, rw_id]).item()
            lift_now = bc_obj_max_z - bc_obj_rest_z
            print(f"[demo] t={t:5.1f}s M3 CUT at bc t={int(bc_clock)}: grasp frozen "
                  f"(grip {grip:.3f} m, lift {lift_now * 100:+.1f} cm, obj_z "
                  f"{target_obj.data.root_pos_w[0, 2].item():.3f}) - backing off, then "
                  f"walking the staged route to the table-B dock ({m3_place_desc})")
            if lift_now < BC_REAL_LIFT_MIN:
                print(f"[demo] WARNING: peak lift {lift_now * 100:+.1f} cm < "
                      f"{BC_REAL_LIFT_MIN * 100:.0f} cm - the pick leg likely FAILED; "
                      f"walking on anyway so the whole pipeline can be observed")

        # ==================================================================
        # M2：BC 策略步（上半身：右臂 + 双手）
        # ==================================================================
        if phase == "BC":
            # --- M3 第二腿入口：让"同一个"时钟在切割 tick 处续跑；放置目标改成"真实的"放置区中心
            # （世界系）——停靠已经把它摆到了训练时的相对位姿上。
            # last_action 里还是切割 tick 的动作、手臂还保持着切割 tick的姿态——反馈是连续的。
            if args_cli.carry_to_b and m3_leg == "carry":
                m3_leg = "place"
                m3_carry_hold = False   # 从这里起BC每步都写目标
                # 关闭搬运握紧：放置腿按普通 base_fix 增益跑，与训练/采集时完全一致
                for jid, kp1, kd1 in grip_gain_refs:
                    override_joint_gains(robot, jid, kp1, kd1)
                # 放置目标 = 放置区中心（世界系），z = 桌面目标高 0.84
                bc_place_target_w = torch.tensor(
                    [env.scene.env_origins[0, 0].item() + place_zone_xy[0],
                     env.scene.env_origins[0, 1].item() + place_zone_xy[1],
                     env.scene.env_origins[0, 2].item() + 0.84 + SURFACE_LIFT],
                    device=device)
                # 诊断打印：放置目标相对右腕的向量，旋到 BC 训练系
                rh2 = robot.data.body_pos_w[0, rw_id]
                dyaw2 = BC_YAW - yaw
                cd2, sd2 = math.cos(dyaw2), math.sin(dyaw2)
                pvx = bc_place_target_w[0].item() - rh2[0].item()
                pvy = bc_place_target_w[1].item() - rh2[1].item()
                prel = (cd2 * pvx - sd2 * pvy, sd2 * pvx + cd2 * pvy,
                        bc_place_target_w[2].item() - rh2[2].item())
                print(f"[demo] t={t:5.1f}s M3 PLACE HANDOVER: BC clock resumes at "
                      f"t={int(bc_clock)}, place target = {m3_place_desc} centre\n"
                      f"[demo]   place_rel(bc) @resume: ({prel[0]:+.3f},{prel[1]:+.3f},"
                      f"{prel[2]:+.3f}) - training-time CARRY-phase place_rel is the "
                      f"reference; base z={robot.data.root_pos_w[0, 2].item():.3f} "
                      f"yaw_err={math.degrees(goal_yaw_err):+.1f} deg")
            # --- 一次性交接初始化（第一个 BC 步，两条入口路径共用） ---
            if bc_place_target_w is None:
                # 记录当前 21 关节位置，作为混合切入的起点
                bc_q_from = robot.data.joint_pos[:, bc_write_ids].clone()
                # 分阶段 / 抬臂专家入场：已经在起始位 -> 不需要混合。
                # 立即接管的网络入场：从 0 混合到 1。
                bc_blend_frac = 0.0 if args_cli.bc_immediate else 1.0
                bc_clock = float(args_cli.bc_start_tick)   # 0 = 回合开始
                # 记录工件静置高度，用于之后的"峰值抬升"判据
                bc_obj_rest_z = target_obj.data.root_pos_w[0, 2].item()
                bc_obj_max_z = bc_obj_rest_z
                if expert is not None:
                    expert.capture_start_pose()
                # 放置目标：围绕"实时"工件位置按类别取盒内随机偏移，机器人系 -> 世界系用"实时"朝向变换
                # 并执行 >=3 cm 位移规则，与 bc_play 完全一致。z 是训练时 0.84 的桌面目标高。
                r_lo, r_hi, f_lo, f_hi = BC_PLACE_BOXES[cur_cls]
                r_off = random.uniform(r_lo, r_hi)      # 右向随机偏移
                f_off = random.uniform(f_lo, f_hi)      # 前向随机偏移
                d_off = max(math.hypot(r_off, f_off), 1e-6)
                if d_off < 0.03:
                    # 位移不足 3 cm：按比例放大到 3 cm（训练时的最小位移规则）
                    r_off, f_off = r_off * 0.03 / d_off, f_off * 0.03 / d_off
                p = target_obj.data.root_pos_w[0]
                yc, ys = math.cos(yaw), math.sin(yaw)   # 前向=(yc,ys)，右向=(ys,-yc)
                # 训练时目标 z ~ 桌面高（0.84）；叠加 SURFACE_LIFT 使"放置目标相对手"的观测保持在抬高后的表面上
                bc_place_target_w = torch.tensor(
                    [p[0].item() + f_off * yc + r_off * ys,
                     p[1].item() + f_off * ys - r_off * yc,
                     env.scene.env_origins[0, 2].item() + 0.84 + SURFACE_LIFT], device=device)
                if args_cli.bc_statue:
                    # 雕像模式（需显式开启；设为默认时会摔倒，见 ）：合成能复现"此刻"站姿的常量动作
                    # （目标 = 默认位 + 0.25*a）。
                    bc_freeze_action = (
                        robot.data.joint_pos[:, walk_act_ids]
                        - robot.data.default_joint_pos[:, walk_act_ids]
                    ) / WALK_ACTION_SCALE
                pitch0, roll0 = quat_pitch_roll(qw, qx, qy, qz)
                # 工件在机器人系下的位置——"交接几何检查"本尊：读数必须约等于(前向 dock_ahead, 侧向 -dock_right)，
                # 否则BC回放一开始就在训练分布之外
                odx0 = p[0].item() - robot.data.root_pos_w[0, 0].item()
                ody0 = p[1].item() - robot.data.root_pos_w[0, 1].item()
                # 交接时的 rel(bc) 对比训练起始状态——"放行/禁行"的几何
                
                rh0 = robot.data.body_pos_w[0, rw_id]
                dyaw0 = BC_YAW - yaw
                cd0, sd0 = math.cos(dyaw0), math.sin(dyaw0)
                rvx, rvy = p[0].item() - rh0[0].item(), p[1].item() - rh0[1].item()
                rel0 = (cd0 * rvx - sd0 * rvy, sd0 * rvx + cd0 * rvy,
                        p[2].item() - rh0[2].item())
                # 交接元数据：随 --bc_ref_dump 一起存盘，供离线对比
                bc_handover_meta = {
                    "rel_bc": rel0,
                    "base_z": robot.data.root_pos_w[0, 2].item(),
                    "pitch_deg": math.degrees(pitch0), "roll_deg": math.degrees(roll0),
                    "yaw_err_deg": math.degrees(goal_yaw_err),
                    "piece_fwd": yc * odx0 + ys * ody0, "piece_lat": -ys * odx0 + yc * ody0,
                    "start_tick": args_cli.bc_start_tick,
                }
                print(f"[demo] t={t:5.1f}s BC HANDOVER: target {cur_cls}, "
                      f"place offset (right {r_off:+.3f}, fwd {f_off:+.3f}), "
                      f"running {args_cli.bc_steps} BC steps "
                      f"({'STATUE legs' if bc_freeze_action is not None else 'walk policy keeps balancing'})\n"
                      f"[demo]   base @handover: z={robot.data.root_pos_w[0, 2].item():.3f} m "
                      f"(BC fixed base 0.760) pitch={math.degrees(pitch0):+.1f} deg "
                      f"roll={math.degrees(roll0):+.1f} deg yaw_err={math.degrees(goal_yaw_err):+.1f} deg\n"
                      f"[demo]   piece @handover: fwd {yc * odx0 + ys * ody0:+.3f} / "
                      f"lat {-ys * odx0 + yc * ody0:+.3f} m "
                      f"(score ref fwd {dock_ahead:+.3f} / lat {-dock_right:+.3f})\n"
                      f"[demo]   rel(bc) @handover: ({rel0[0]:+.3f},{rel0[1]:+.3f},{rel0[2]:+.3f}) "
                      f"- OLD fixed-base reference (+0.108,-0.130,-0.038) and its mid-INSERT "
                      f"dive warning apply to pre-2026-08 checkpoints ONLY; raised-start "
                      f"datasets start at THIS pose by construction (piece-specific)")
            # --- 手工拼装 110 维观测，镜像 pickplace 的 ObservationsCfg ---
            jp = robot.data.joint_pos[0].clone()
            jv = robot.data.joint_vel[0].clone()
            # 腿+腰在 BC 环境里被"冻结在零位"；这里也钳到该训练分布，而不是把实际站姿泄漏进观测
            jp[bc_frozen_ids] = 0.0
            jv[bc_frozen_ids] = 0.0
            # 左臂：在这里归行走策略管，但网络训练时它被"钉"在保持位、
            # 速度约为零——同样钳到那个常量（见 BC_LEFT_ARM_OBS_POSE）
            jp[bc_left_ids] = bc_left_pose
            jv[bc_left_ids] = 0.0
            # 世界系 -> BC 训练世界系的旋转，用于各世界系相对向量
            dyaw = BC_YAW - yaw
            cd, sd = math.cos(dyaw), math.sin(dyaw)

            def to_bc_frame(v: torch.Tensor) -> torch.Tensor:
                #把世界系向量绕 z 轴旋转 dyaw，得到 BC 训练系下的向量。
                return torch.stack((cd * v[0] - sd * v[1], sd * v[0] + cd * v[1], v[2]))

            obj_p = target_obj.data.root_pos_w[0]
            bc_obj_max_z = max(bc_obj_max_z, obj_p[2].item())  # 跟踪峰值高度（抬升判据）
            lh_p = robot.data.body_pos_w[0, lw_id]             # 左腕位置
            rh_p = robot.data.body_pos_w[0, rw_id]             # 右腕位置
            # 相位时钟：0..1，告诉网络回合进行到哪一步了
            phase_frac = min(bc_clock, float(BC_MAX_EPISODE_STEPS)) / BC_MAX_EPISODE_STEPS
            # 110 维 = 15 体关节位置 + 15 体关节速度 + 14 手关节位置
            #        + 物体相对左腕 3 + 物体相对右腕 3 + 上一步动作 28
            #        + 相位 1 + 放置目标相对右腕 3
            obs_bc = torch.cat([
                jp[bc_body_idx], jv[bc_body_idx], jp[bc_hand_idx],
                to_bc_frame(obj_p - lh_p), to_bc_frame(obj_p - rh_p),
                bc_last_action[0],
                torch.tensor([phase_frac], device=device),
                to_bc_frame(bc_place_target_w - rh_p),
            ]).unsqueeze(0)
            dagger_label = None
            if args_cli.bc_dagger:
                # DAgger：由"网络"开车，因此访问到的状态是它"自己的"闭环分布（含漂移）；专家对每个状态给出纠正性标注
                with torch.no_grad():
                    bc_action = bc_policy(obs_bc)
                dagger_label = expert.action28(bc_clock, bc_place_target_w)
            elif expert is not None:
                # 脚本化专家开车跑整个周期：同一套观测管线、同一种动作编码（原始 a，目标 = 0.5*a）、
                # 同一条写入路径——因此录下的(obs, action)行"就是"重训后 BC 将见到的部署分布
                bc_action = expert.action28(bc_clock, bc_place_target_w)
            else:
                # 纯网络推理
                with torch.no_grad():
                    bc_action = bc_policy(obs_bc)
            if expert_rows is not None and bc_clock < args_cli.bc_steps:
                # 数据集采集：DAgger 存专家标注，否则存执行的动作
                label = dagger_label if dagger_label is not None else bc_action
                expert_rows.append((obs_bc[0].detach().cpu().clone(),
                                    label[0].detach().cpu().clone()))
            if bc_dump_rows is not None and not bc_dump_saved:
                # --dump_obs：录前 200 tick 的 obs/action 供离线诊断
                if bc_clock < args_cli.bc_start_tick + 200:
                    bc_dump_rows.append((bc_clock, obs_bc[0].detach().cpu().clone(),
                                         bc_action[0].detach().cpu().clone()))
                else:
                    bc_dump_saved = True
                    torch.save({
                        "source": "transport_demo",
                        "workpiece": args_cli.target,
                        "checkpoint": args_cli.bc_checkpoint,
                        "body_idx": BC_BODY_OBS_IDX,
                        "hand_idx": BC_HAND_OBS_IDX,
                        "body_names": [robot.joint_names[i] for i in BC_BODY_OBS_IDX],
                        "hand_names": [robot.joint_names[i] for i in BC_HAND_OBS_IDX],
                        "handover": bc_handover_meta,
                        "tick": [row[0] for row in bc_dump_rows],
                        "obs": torch.stack([row[1] for row in bc_dump_rows]),
                        "act": torch.stack([row[2] for row in bc_dump_rows]),
                    }, args_cli.dump_obs)
                    print(f"[demo] dumped {len(bc_dump_rows)} BC obs/action rows "
                          f"-> {args_cli.dump_obs}")
            # 目标写入会持续到 --bc_steps 之后，供专家周期结束后的"回抬臂"动作使用
            # DAgger 执行的是"网络"，它从没被训练到 tick830 之后——就冻结在那里。
            bc_apply_until = args_cli.bc_steps
            if expert is not None and not args_cli.bc_dagger:
                bc_apply_until += BC_RETURN_SETTLE + BC_RETURN_TICKS + 20
            if bc_clock < bc_apply_until:
                # DART 风格执行噪声（--action_noise）：扰动加在"写入的"目标上；录下的标注保持"干净"
                # last_action 观测反映"执行的"动作，与 grasp-expert
                # 环境侧的记账方式一致（环境存的是实际步进的带噪动作）。
                exec_action = bc_action
                if (expert is not None and args_cli.action_noise > 0.0
                        and bc_clock < args_cli.bc_steps):
                    exec_action = bc_action + args_cli.action_noise * torch.randn_like(bc_action)
                # 目标 = BC 环境默认位（所有被写关节都是 0）+ 0.5*a
                bc_targets = (BC_ACTION_SCALE * exec_action)[:, bc_write_slots_t]
                # 立即接管的混合：把写入的目标从交接姿态平滑过渡到 BC 指令（--bc_blend）；
                # 默认分阶段交接下是空操作（frac 从1开始）
                bc_blend_frac = min(1.0, bc_blend_frac + CTRL_DT / max(args_cli.bc_blend, CTRL_DT))
                robot.set_joint_position_target(
                    bc_q_from + bc_blend_frac * (bc_targets - bc_q_from),
                    joint_ids=bc_write_ids)
                bc_last_action = exec_action.clone()   # 下一步观测里的 last_action
                # 早期细粒度遥测
                if bc_clock < 160 and int(bc_clock) % 20 < BC_CLOCK_PER_STEP:
                    rel_e = to_bc_frame(obj_p - rh_p)
                    print(f"[demo]   bc t={int(bc_clock):3d} "
                          f"elb tgt={bc_targets[0, bc_elb_write_idx].item():+.2f}"
                          f"/act={robot.data.joint_pos[0, elbow_jid].item():+.2f} "
                          f"blend={bc_blend_frac:.2f} "
                          f"wrist_z={rh_p[2].item():.3f} "
                          f"rel=({rel_e[0].item():+.3f},{rel_e[1].item():+.3f},{rel_e[2].item():+.3f})")
            if bc_clock >= args_cli.bc_steps and not bc_report_done:
                # 到达周期末尾（回抬臂可能还在写目标）；按 bc_play 的方式打分
                bc_report_done = True
                d_xy = math.hypot(obj_p[0].item() - bc_place_target_w[0].item(),
                                  obj_p[1].item() - bc_place_target_w[1].item())
                # 停靠系下的误差"向量"（2026-08-13）：第二轮螺母回归证明
                # 了释放偏移的"方向"很重要，只看幅值没法指导修正。
                # + fwd = 沿机器人朝向越过了目标，+ left = 在目标的机器人
                # 左侧。
                _dxw = obj_p[0].item() - bc_place_target_w[0].item()
                _dyw = obj_p[1].item() - bc_place_target_w[1].item()
                d_fwd = math.cos(yaw) * _dxw + math.sin(yaw) * _dyw
                d_left = -math.sin(yaw) * _dxw + math.cos(yaw) * _dyw
                # dz：物体相对静置高度的变化（扣除类别的放置根高差）
                dz = obj_p[2].item() - bc_obj_rest_z - BC_PLACE_ROOT_DZ[cur_cls]
                spd = torch.norm(target_obj.data.root_lin_vel_w[0]).item()
                # 判定通过：水平误差、竖直误差、速度三项都要过门限
                ok = d_xy < BC_PLACE_XY_TOL and abs(dz) < BC_PLACE_Z_TOL and spd < BC_PLACE_SPEED_TOL
                # 真正的抓取-放置会先把工件"抬起来"（bc_play 方块参考：
                # obj_z 0.824 -> 0.920，LIFT/CARRY 阶段 +9.6 cm
                lift = bc_obj_max_z - bc_obj_rest_z
                verdict = "PLACED" if ok else "NOT PLACED"
                if ok and lift < BC_REAL_LIFT_MIN:
                    verdict = "PLACED-BY-PUSH (never lifted!)"
                print(f"[demo] ===== M2 BC REPORT ({verdict}) =====\n"
                      f"[demo]   obj->target xy: {d_xy * 100:5.1f} cm (gate {BC_PLACE_XY_TOL * 100:.0f} cm)\n"
                      f"[demo]   obj->target vec: fwd {d_fwd * 100:+5.1f} cm, left {d_left * 100:+5.1f} cm (dock frame)\n"
                      f"[demo]   obj dz vs rest: {dz * 100:+5.1f} cm (gate +-{BC_PLACE_Z_TOL * 100:.0f} cm)\n"
                      f"[demo]   obj max lift  : {lift * 100:+5.1f} cm (bc_play grasp reference +9.6 cm)\n"
                      f"[demo]   obj speed     : {spd:.3f} m/s (gate {BC_PLACE_SPEED_TOL:.2f})\n"
                      f"[demo]   (upright/grasp quality: check the GUI)")
                if args_cli.loop_all:
                    m4_records.append((len(m4_results), cur_cls, m3_place_desc,
                                       verdict, d_xy * 100.0, retry_count,
                                       d_fwd * 100.0, d_left * 100.0))
                    m4_results.append(
                        f"job {m4_job_idx + 1}/{len(m4_jobs)} {cur_cls} -> "
                        f"{m3_place_desc}: {verdict} "
                        f"(obj->target {d_xy * 100:.1f} cm, lift {lift * 100:+.1f} cm)")
                    if args_cli.repeat > 1 and len(m4_results) >= len(m4_jobs):
                        m4_iter_end = True  # 最后一个作业：迭代结束
                if expert_rows is not None:
                    # 专家采集：只有干净的完整周期才进数据集（PLACED +真实抬升，与 grasp-expert.py 相同）。
                    # DAgger：保留"每一个"回合——网络自己的漂移状态就是训练信号，而标注是专家给的、干净的。
                    if (ok and lift >= BC_REAL_LIFT_MIN) or args_cli.bc_dagger:
                        obs_np = torch.stack([r[0] for r in expert_rows]).numpy()
                        act_np = torch.stack([r[1] for r in expert_rows]).numpy()
                        if os.path.exists(args_cli.save_dataset):
                            # 已有数据集：追加而不是覆盖
                            prev = np.load(args_cli.save_dataset)
                            obs_np = np.concatenate([prev["obs"], obs_np], axis=0)
                            act_np = np.concatenate([prev["actions"], act_np], axis=0)
                        os.makedirs(os.path.dirname(args_cli.save_dataset) or ".", exist_ok=True)
                        np.savez(args_cli.save_dataset, obs=obs_np, actions=act_np)
                        tag = "DAgger" if args_cli.bc_dagger else "expert"
                        print(f"[demo] {tag} dataset: +{len(expert_rows)} rows "
                              f"(file total {obs_np.shape[0]}) -> {args_cli.save_dataset}")
                    else:
                        print("[demo] expert cycle did not score PLACED+lift - "
                              "rows NOT appended to the dataset")
            # 相位边界遥测：与bc_play可1:1对比，另加逐轴分解，用来看
            # "哪个方向"在发散：rel(bc)=物体-右腕，在BC训练系下
            # （x=BC 世界的 x ~ 机器人的"左"，y 为负 = 前方，z 向上）。
            for mark, mname in BC_PHASE_MARKS.items():
                if bc_clock <= mark < bc_clock + BC_CLOCK_PER_STEP:
                    rel_bc = to_bc_frame(obj_p - rh_p)
                    pitch_m, roll_m = quat_pitch_roll(qw, qx, qy, qz)
                    if dagger_label is not None:
                        # 网络与专家在"这个状态"上的差距——DAgger 的纠正就集中在差距大的地方
                        dev = (bc_action - dagger_label).abs().max().item()
                        print(f"[demo]   dagger t={mark:3d} max|net-expert| = {dev:.3f}")
                    print(f"[demo]   bc t={mark:3d} after {mname:7s} "
                          f"hand->obj={torch.norm(obj_p - rh_p).item():.3f} m "
                          f"rel(bc)=({rel_bc[0].item():+.3f},{rel_bc[1].item():+.3f},{rel_bc[2].item():+.3f}) | "
                          f"wrist_z={rh_p[2].item():.3f} obj_z={obj_p[2].item():.3f} | "
                          f"obj->target xy={math.hypot(obj_p[0].item() - bc_place_target_w[0].item(), obj_p[1].item() - bc_place_target_w[1].item()):.3f} m | "
                          f"base z={robot.data.root_pos_w[0, 2].item():.3f} "
                          f"p/r=({math.degrees(pitch_m):+.1f},{math.degrees(roll_m):+.1f}) deg")
            # 相位推进：宿主 50 Hz，每步 +2 tick 对齐 100 Hz 的训练时钟
            bc_clock += BC_CLOCK_PER_STEP
            # 采集模式：周期出分且周期后的回抬 ramp 结束后就退出，让外层 shell 循环能串联回合，不用空等到--max_time。
            if (args_cli.save_dataset and bc_report_done
                    and bc_clock >= bc_apply_until):
                print("[demo] collection episode finished - exiting")
                break
            # M4 循环：报告出完 + 短暂静置 -> 要么抬臂走回去干下一个作业，要么在最后一个作业后收尾。
            if (args_cli.loop_all and bc_report_done
                    and bc_clock >= args_cli.bc_steps + M4_REPORT_HOLD_TICKS):
                phase = "ARMRET"
                armret_frac = 0.0
                armret_from = robot.data.joint_pos[:, arm_ids].clone()
                m3_carry_hold = True    # ARMRET 期间手臂目标归它管
                last_job = m4_job_idx + 1 >= len(m4_jobs)
                print(f"[demo] t={t:5.1f}s job {m4_job_idx + 1}/{len(m4_jobs)} placed - "
                      f"raising the arm back to the walk pose "
                      f"({M4_ARM_RETURN_SEC:.1f}s), then "
                      f"{'walking back to the start point' if last_job else 'returning to table A'}")
            # M3 单次搬运：报告已出、网络在 bc_steps 之后被冻结——为 GUI多停几秒后结束，不空等到 --max_time。
            elif (args_cli.carry_to_b and not args_cli.loop_all and bc_report_done
                    and bc_clock >= args_cli.bc_steps + 500):
                print("[demo] M3 transport cycle complete - exiting")
                break

        # ---- --repeat：迭代边界（所有作业都有了判定） ----
        if m4_iter_end:
            m4_iter_end = False
            it_recs = m4_records[-len(m4_jobs):]    # 本轮迭代的记录
            verd = ", ".join(r[3] if r[4] is None else f"{r[3]} ({r[4]:.1f} cm)"
                             for r in it_recs)
            print(f"[demo] ===== --repeat: iteration {m4_iter}/{args_cli.repeat} "
                  f"done in {t - iter_t0:.0f} s: {verd} =====", flush=True)
            if m4_stats_csv:
                # 每轮迭代都追加+关闭文件：这样Isaac退出时崩溃、吞掉管道里的控制台最后数据，统计也能存活
                hdr = not os.path.exists(m4_stats_csv)
                with open(m4_stats_csv, "a", encoding="utf-8") as fcsv:
                    if hdr:
                        fcsv.write("iter,job,cls,spot,verdict,obj_target_cm,retries,"
                                   "err_fwd_cm,err_left_cm\n")
                    for r in it_recs:
                        fcsv.write(f"{m4_iter},{r[0] + 1},{r[1]},{r[2]},{r[3]},"
                                   f"{'' if r[4] is None else round(r[4], 1)},{r[5]},"
                                   f"{'' if r[6] is None else round(r[6], 1)},"
                                   f"{'' if r[7] is None else round(r[7], 1)}\n")
            if m4_iter >= args_cli.repeat:
                # 全部迭代跑完：出总结并结束
                m4_batch_summary()
                m4_summary_done = True
                m4_done = True
                break
            m4_iter += 1
            m4_batch_reset()            # 彻底复位，开始下一轮
            continue

        # ---- 行走策略推理 + 环境步进（manager 管线，与训练一致） ----
        with torch.inference_mode():
            action = policy(obs)        # PPO：69 维观测 -> 19 维腿/腰/左臂动作
        if phase == "BC" and bc_freeze_action is not None:
            # 雕像模式：绕过行走策略；动作管理器把腿 + 左臂 PD 保持在交接时的站姿
            action = bc_freeze_action
        obs_dict = env.step(action)[0]  # 物理步进 + 重新计算观测
        obs = obs_dict["policy"]
        t += CTRL_DT

        # ---- --record：把调试相机的帧流式写盘 ----
        # 时间累加器采样：仿真时间每跨过一个 1/REC_FPS 边界就取一帧。
        # 30 Hz 不能整除 50 Hz 的控制频率，固定每 N 步取一帧做不到 1 倍速的 30 fps——这个方法可以。
        if rec_sinks is not None:
            rec_acc += CTRL_DT
            if rec_acc >= 1.0 / REC_FPS:
                rec_acc -= 1.0 / REC_FPS
                for name, sink in rec_sinks.items():
                    sink.write(_cam_frame(env.scene.sensors[name]))

        # ---- 1 Hz 遥测打印 ----
        if t - last_report >= 1.0:
            last_report = t
            c = cmd_term.vel_command_b[0]
            # 手掌前伸量是"与碰撞相关"的关键数：停靠时桌面边缘就在骨盆前方约 0.13 m 处。
            pdx = robot.data.body_pos_w[0, palm_id, 0].item() - robot.data.root_pos_w[0, 0].item()
            pdy = robot.data.body_pos_w[0, palm_id, 1].item() - robot.data.root_pos_w[0, 1].item()
            palm_fwd = math.cos(yaw) * pdx + math.sin(yaw) * pdy
            # 目标工件在机体系下的位置；按工件的 BC 位姿，停靠后应为
            # (前向 dock_ahead, 侧向 -dock_right)——lat<0 = 机器人右侧
            obj_txt = ""
            if target_obj is not None:
                odx = target_obj.data.root_pos_w[0, 0].item() - robot.data.root_pos_w[0, 0].item()
                ody = target_obj.data.root_pos_w[0, 1].item() - robot.data.root_pos_w[0, 1].item()
                obj_txt = (f"obj=(fwd {math.cos(yaw) * odx + math.sin(yaw) * ody:+.2f}, "
                           f"lat {-math.sin(yaw) * odx + math.cos(yaw) * ody:+.2f}) ")
                if m3_leg == "carry":
                    # 滑落监视：整个行走过程中 手->物 距离应保持约等于、抓取偏移（手腕到工件 0.15-0.20 m）
                    grip = torch.norm(target_obj.data.root_pos_w[0]
                                      - robot.data.body_pos_w[0, rw_id]).item()
                    obj_txt += f"grip={grip:.2f} m obj_z={target_obj.data.root_pos_w[0, 2].item():.2f} "
            print(f"[demo] t={t:5.1f}s {phase:5s} dist={dist:5.2f} m "
                  f"head_err={math.degrees(head_err):6.1f} goalyaw_err={math.degrees(goal_yaw_err):6.1f} "
                  f"cmd=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f}) "
                  f"h={robot.data.root_pos_w[0, 2].item():.2f} arm=(raise {raise_frac:.1f}, bc {bc_frac:.1f}) "
                  # 肘 tgt/act：若实际滞后目标 >0.1 rad，说明手臂 PD 被重力压制、需要更大的 --raise_elbow；
                  # 若实际跟得上目标但手掌 z 仍然偏低，则是姿态模型本身错了。
                  f"elb={q_arm_target[0, elbow_slot].item():+.2f}/{robot.data.joint_pos[0, elbow_jid].item():+.2f} "
                  f"{obj_txt}palm=(fwd {palm_fwd:+.2f}, z {robot.data.body_pos_w[0, palm_id, 2].item():.2f})")

    # ---- 收尾：判断是正常完成还是超时 ----
    finished = (phase == "DONE" or m4_done
                or (phase == "BC" and bc_active and bc_report_done))
    if not finished:
        print(f"[demo] TIMEOUT in phase {phase} after {t:.0f}s - see telemetry above")
        if args_cli.loop_all and m4_results:
            print(f"[demo] M4 jobs completed before the timeout:")
            for line in m4_results:
                print(f"[demo]   {line}")
    if args_cli.repeat > 1 and m4_records and not m4_summary_done:
        # 批量被提前打断（--max_time 到 / 窗口被关）：仍然对已完成的部分报告百分比
        m4_batch_summary()
    if rec_sinks is not None:
        for name, sink in rec_sinks.items():
            sink.close()
            print(f"[demo] video saved: {sink.path} "
                  f"({sink.count} frames, {sink.backend} backend)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
