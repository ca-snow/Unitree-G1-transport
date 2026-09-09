# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
测量 USD 资产的包围盒尺寸、单位制、以及已挂载的物理 API。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure USD assets.")
# 位置参数 paths：一个或多个 USD 文件路径（本地路径或 URL），nargs="+" 表示至少一个
parser.add_argument("paths", nargs="+", help="USD file paths (or URLs).")
# 挂上 Isaac Lab 启动器参数（--headless 等）
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim（需要它是因为 pxr/USD 库和资产解析随 Isaac 一起提供）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr 是 Pixar USD 的 Python 绑定：Usd=场景文件，UsdGeom=几何，UsdPhysics=物理 schema
from pxr import Usd, UsdGeom, UsdPhysics


def measure(path: str):
    #打开一个 USD 文件，打印它的尺寸和物理属性信息。
    #出错时只打印错误并 return，不抛异常——一个坏路径不能中断后面其余资产的测量。
    print(f"\n=== {path} ===")
    try:
        stage = Usd.Stage.Open(path)
    except Exception as exc:
        print(f"  [ERR] cannot open: {exc}")
        return
    if stage is None:
        print("  [ERR] cannot open (check the path with ls)")
        return

    # metersPerUnit：1个 USD 单位等于多少米（1.0=米制，0.01=厘米制）后面所有尺寸都要乘它换算成米
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)  # 向上轴（Y 或 Z），Isaac 要求 Z 向上
    # 取默认 prim（资产的根节点）；没有默认prim就退回到伪根（整个场景树的顶）
    prim = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    # 用缓存计算世界坐标；purpose 里带上 default 和 render
    # 两类几何（有些资产的可见网格标记为 render purpose，漏掉会测出）
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        print("  [ERR] empty bbox (no geometry under the default prim?)")
        return
    size = box.GetMax() - box.GetMin()  # 三边长（USD 单位）
    size_m = [s * mpu for s in size]    # 换算成米

    print(f"  metersPerUnit={mpu}  upAxis={up}  defaultPrim={prim.GetPath()}")
    print(f"  bbox size    = {size_m[0]:.4f} x {size_m[1]:.4f} x {size_m[2]:.4f} m")
    # min/max 也打出来：如果几何中心不在原点（原点偏移），生成（spawn）时的
    # z 坐标要按偏移量修正，否则物体会陷进桌面或悬空
    print(f"  bbox min/max = {[round(v * mpu, 4) for v in box.GetMin()]} .. "
          f"{[round(v * mpu, 4) for v in box.GetMax()]} (origin offset matters for spawn z)")

    # 遍历整个场景树，统计物理 API 的挂载情况
    rb, col, mass_prims = [], 0, []
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb.append(str(p.GetPath()))       # 挂了刚体 API 的 prim 路径
        if p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.MeshCollisionAPI):
            col += 1                          # 碰撞体 prim 计数
        if p.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(p).GetMassAttr().Get()
            mass_prims.append((str(p.GetPath()), m))  # 资产里写死的质量
    # 若为 NONE：资产不带刚体/碰撞体，接入场景时必须在配置里补
    # rigid_props / collision_props（或先用 prepare_*_asset.py 预处理）
    print(f"  RigidBodyAPI on: {rb if rb else 'NONE (must add rigid_props in the cfg)'}")
    print(f"  collision prims: {col if col else 'NONE (must add collision_props in the cfg)'}")
    for pth, m in mass_prims:
        print(f"  authored mass  : {m} kg on {pth}")

# 主流程：对命令行传入的每个路径依次测量
for p in args_cli.paths:
    measure(p)

simulation_app.close()
