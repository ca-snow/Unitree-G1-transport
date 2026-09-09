# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
把"只有外观、没有物理"的 USD 资产（YCB 电钻）加工成可参与物理仿真的新 USD。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Author physics onto a visual-only USD asset.")
# --src：只有外观的源 USD。默认是 Isaac Sim 5.1 官方资产库里的 YCB 电钻
parser.add_argument(
    "--src", type=str,
    default="/opt/NVIDIA/isaacsim-assets/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd",
    help="Visual-only source USD.")
# --dst：输出 USD 路径（已存在则覆盖）。默认写到项目自己的 assets/objects 目录
parser.add_argument(
    "--dst", type=str,
    default="/home/0000_wy/unitree/unitree_sim_isaaclab/assets/objects/035_power_drill_physics.usd",
    help="Output USD path (overwritten if it exists).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim（USD/物理 schema 库随它一起提供）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from pxr import Usd, UsdGeom, UsdPhysics


def main():
    #新建输出 USD：引用源资产，在根节点补刚体/质量 API，给所有网格加凸分解碰撞体。
    # 确保输出目录存在；旧文件先删掉，保证是干净的新建（CreateNew 遇到已有文件会报错）
    os.makedirs(os.path.dirname(args_cli.dst), exist_ok=True)
    if os.path.exists(args_cli.dst):
        os.remove(args_cli.dst)

    stage = Usd.Stage.CreateNew(args_cli.dst)
    # 单位与朝向和源资产一致：实测源资产就是"米制、Z 轴向上"，这里显式写死
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    # 建一个 /Object 根节点，通过"引用"挂进源资产——不复制几何数据，
    # 输出文件很小，且源资产更新后这里自动跟随
    root = UsdGeom.Xform.Define(stage, "/Object")
    prim = root.GetPrim()
    prim.GetReferences().AddReference(args_cli.src)
    stage.SetDefaultPrim(prim)  # 设为默认 prim，别人引用本文件时以它为根

    # 根节点补上刚体和质量 API（这是源资产缺失的关键物理属性）
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)  # 只挂 API 占位；真实质量数值由场景配置覆盖

    # 遍历（穿透引用）找出所有网格，逐个挂碰撞体
    n_mesh = 0
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(p)
            mesh_col = UsdPhysics.MeshCollisionAPI.Apply(p)
            # 碰撞近似方式选"凸分解"：把非凸网格拆成多个小凸块，贴合真实表面
            mesh_col.CreateApproximationAttr().Set("convexDecomposition")
            n_mesh += 1

    stage.Save()
    print(f"[prepare] wrote {args_cli.dst}")
    print(f"[prepare]   RigidBodyAPI + MassAPI on {prim.GetPath()}")
    print(f"[prepare]   convex-decomposition colliders on {n_mesh} mesh(es)")
    if n_mesh == 0:
        # 一个网格都没找到：源资产可能用了"实例化 prim"（Traverse 默认进不去），需要人工排查
        print("[prepare]   WARNING: no meshes found - the source may use instanced prims; report this.")


main()
simulation_app.close()
