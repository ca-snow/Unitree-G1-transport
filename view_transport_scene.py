# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""双桌搬运（TRANSPORT）场景的交互式查看器（不训练、也不走 gym 接口）。

加载 tasks/rl/g1_transport/transport_scene_cfg.py 的场景配置，只开 1 个环境、
单纯跑物理仿真，用于肉眼检查和调整场景布局（两张桌子的位置、放置区标记、
机器人初始站位等）。窗口开着的时候，还可以顺便打开 Isaac Sim 资产库
（菜单：Window > Browsers > Isaac Sim Assets），把候选工件拖到 A 桌上、
与红色参考立方体（边长 0.06 米）比大小——选好后把资产的名字/路径记下来上报。

"""

import argparse
import sys

from isaaclab.app import AppLauncher

# 本脚本自己不需要额外参数，只挂 Isaac Lab 启动器的通用参数（--device 等）
parser = argparse.ArgumentParser(description="View the G1 transport scene.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim（必须先于其他 isaaclab import）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

# 云端项目根目录：场景配置在 import 时读 PROJECT_ROOT 拼资产路径，必须先设好
PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# TransportSceneCfg：场景配置类；layout：同一模块，取里面的布局常量（桌子坐标等）
from tasks.rl.g1_transport import TransportSceneCfg
from tasks.rl.g1_transport import transport_scene_cfg as layout


def main():
    #搭建仿真上下文和场景，打印布局信息，然后进入"站立不动"的物理循环。
    # 开启 CCD（连续碰撞检测），与训练环境保持一致：螺栓是立在一小块很薄的
    # 接触面上的，落桌沉降时如果不开 CCD 会直接"穿隧"（tunneling）穿过桌面。
    # dt=0.005：物理步长 5 毫秒（200Hz），与训练环境相同。
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=0.005, device=args_cli.device or "cuda:0",
        physx=sim_utils.PhysxCfg(enable_ccd=True)))
    # 相机取"俯视 3/4 视角"：架在两桌中点上方偏南，一屏看全两张桌子和中间的行走通道
    mid_x = (layout.TABLE_A_POS[0] + layout.TABLE_B_POS[0]) / 2.0
    sim.set_camera_view(eye=(mid_x, -7.5, 6.0), target=(mid_x, -0.7, 0.8))

    # 只建 1 个环境；env_spacing=20.0 米的间距在单环境下无实际影响，仅为参数完整
    scene = InteractiveScene(TransportSceneCfg(num_envs=1, env_spacing=20.0))
    sim.reset()  # 初始化物理，把所有资产真正生成到仿真里

    robot = scene["robot"]
    print("[view] scene ready.")
    # 把布局关键量打印出来，方便与画面对照检查：
    # A 桌中心（长边沿 y 轴，从 +x 方向接近）/ B 桌中心（长边沿 x 轴，从 +y 方向接近）
    print(f"[view]   table A centre : {layout.TABLE_A_POS[:2]} (long axis along y, approach from +x)")
    print(f"[view]   table B centre : {layout.TABLE_B_POS[:2]} (long axis along x, approach from +y)")
    # A 桌上工件的随机出生区（中心 + 尺寸）
    print(f"[view]   spawn zone A   : {layout.SPAWN_ZONE_CENTER} size {layout.SPAWN_ZONE_SIZE}")
    # B 桌上的三个放置区（蓝/绿/黄三色方块），x 坐标列表 + 公共 y 坐标 + 边长
    print(f"[view]   place zones B  : x={layout.PLACE_ZONE_XS} y={layout.PLACE_ZONE_Y:.2f} "
          f"(blue/green/yellow, {layout.PLACE_ZONE_SIZE} m squares)")
    # 停靠点：机器人骨盆（pelvis）应到达的位置和朝向（A 桌朝 -x，B 桌朝 -y）
    print(f"[view]   dock A (pelvis): {layout.DOCK_A_POS[:2]} facing -x")
    print(f"[view]   docks B        : {[tuple(round(v, 2) for v in p[:2]) for p in layout.DOCK_B_POSES]} facing -y")
    print(f"[view]   robot start    : {layout.ROBOT_START_POS[:2]} (provisional)")
    print("[view] browse assets via Window > Browsers > Isaac Sim Assets; red cube = 0.06 m reference.")

    def report_workpieces(tag: str):
        #打印各工件根刚体的世界坐标。

        #在"刚出生"和"沉降后"各打一次，就能区分两类问题：
        #是一开始就出生在错误位置，还是出生后掉落/被瞬移走了
        
        for name in ("object", "bolt", "nut", "drill"):
            if name in scene.rigid_objects:  # 场景里配了哪个工件就打哪个
                p = scene[name].data.root_pos_w[0]
                print(f"[view]   {tag} {name:6s} pos=({p[0].item():+.3f}, {p[1].item():+.3f}, {p[2].item():+.3f})")

    # ---------- 物理主循环：窗口开着就一直跑 ----------
    sim_dt = sim.get_physics_dt()
    step = 0
    while simulation_app.is_running():
        # 每步都把关节目标钉在默认姿态上，固定基座的机器人就会原地站好不动
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()  # 把目标写进物理引擎
        sim.step()                 # 物理前进一步
        scene.update(sim_dt)       # 回读仿真数据到场景缓冲
        if step == 0:
            report_workpieces("at spawn  ")   # 出生瞬间的位置
        elif step == 400:  # 400 步 × 0.005 秒 ≈ 2 秒，工件应已沉降稳定
            report_workpieces("settled   ")
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()  
