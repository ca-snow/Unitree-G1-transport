# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
剔除 Factory 装配资产（螺栓/螺母）里自带的物理"关节"定义。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Strip physics joints from a factory USD.")
# --src：源资产。默认是 Isaac Sim 5.1 官方 Factory 库里的 M20 螺栓
parser.add_argument(
    "--src", type=str,
    default="/opt/NVIDIA/isaacsim-assets/Assets/Isaac/5.1/Isaac/Props/Factory/factory_bolt_m20_tight/factory_bolt_m20_tight.usd",
    help="Source factory asset USD.")
# --dst：输出路径（已存在则覆盖）。"plain" 后缀表示"剔除关节后的普通刚体版"
parser.add_argument(
    "--dst", type=str,
    default="/home/0000_wy/unitree/unitree_sim_isaaclab/assets/objects/factory_bolt_m20_plain.usd",
    help="Output USD path (overwritten).")
# --high_friction：把 静摩擦10 / 动摩擦1.5 / 弹性0.01 的物理材质烘进输出文件，
# 并绑定到所有碰撞 prim（摩擦合成模式 max、弹性合成模式 min）。螺母需要它，
# 原因见文件顶部说明
parser.add_argument(
    "--high_friction", action="store_true",
    help="Bake a static 10 / dynamic 1.5 / restitution 0.01 physics material "
    "into the output and bind it to every collision prim (combine mode max/min).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim（USD/PhysX schema 库随它一起提供）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


def main():
    #扫描源资产里的关节，生成"引用源资产 + 停用全部关节（可选烘高摩擦材质）"的新 USD。
    src_stage = Usd.Stage.Open(args_cli.src)
    if src_stage is None:
        print(f"[factory] ERROR: cannot open {args_cli.src}")
        return
    # 源资产默认 prim 的路径前缀，后面做路径换算要用
    src_default = str(src_stage.GetDefaultPrim().GetPath())

    # 第一步：在源资产里找出所有物理关节（顺带打印 ArticulationRoot 的位置供参考）
    joint_paths = []
    for prim in src_stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint_paths.append(str(prim.GetPath()))
            print(f"[factory] joint found: {prim.GetPath()}  <{prim.GetTypeName()}>")
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            # 关节树根标记保留不删：生成时已由配置禁用
            print(f"[factory] articulation root on: {prim.GetPath()} (left in place; disabled at spawn)")
    if not joint_paths:
        # 没找到关节：说明"吸回原点"问题另有原因，需要人工看这份输出排查
        print("[factory] WARNING: no joints found - the origin-snap must have another cause; report this output.")

    # 第二步：新建输出 USD（先删旧文件），单位制/向上轴照抄源资产
    os.makedirs(os.path.dirname(args_cli.dst), exist_ok=True)
    if os.path.exists(args_cli.dst):
        os.remove(args_cli.dst)
    stage = Usd.Stage.CreateNew(args_cli.dst)
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(src_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(src_stage))

    # 建 /Object 根节点并"引用"源资产（文件随源更新）
    root = UsdGeom.Xform.Define(stage, "/Object")
    root.GetPrim().GetReferences().AddReference(args_cli.src)
    stage.SetDefaultPrim(root.GetPrim())

    # 第三步：把源资产里的关节路径换算到新场景树下并逐个停用。
    # 换算规则：引用之后，源资产默认 prim 下的内容都挂在 /Object 下，
    # 所以把路径开头的源默认 prim 前缀替换成 "/Object" 即可
    for src_path in joint_paths:
        local_path = "/Object" + src_path[len(src_default):] if src_path.startswith(src_default) else None
        prim = stage.GetPrimAtPath(local_path) if local_path else None
        if prim:
            prim.SetActive(False)  # 停用 = 该 prim（含关节约束）在合成时被剔除
            print(f"[factory] deactivated {local_path}")
        else:
            print(f"[factory] ERROR: cannot resolve {src_path} in the new stage; report this.")

    # 第四步（可选）：烘焙高摩擦物理材质
    if args_cli.high_friction:
        # 数值与 transport_scene_cfg.py 里红色立方体的 RigidBodyMaterialCfg
        # 完全一致（静摩擦10 / 动摩擦1.5 / 弹性0.01）。绑定在根节点上、
        # purpose 设为 "physics"、强度设为 strongerThanDescendants
        # 这样引用进来的每个碰撞 prim 都会继承这份材质，并压过源资产里原本写的任何低摩擦材质。
        mat = UsdShade.Material.Define(stage, "/Object/HighFrictionMaterial")
        mat_api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        mat_api.CreateStaticFrictionAttr().Set(10.0)   # 静摩擦系数：10，特意取很大防滑
        mat_api.CreateDynamicFrictionAttr().Set(1.5)   # 动摩擦系数：1.5
        mat_api.CreateRestitutionAttr().Set(0.01)      # 弹性恢复系数：接近 0，落桌不弹跳
        px_api = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        # 两个物体接触时的参数合成方式：摩擦取双方较大值（保证抓得住），
        # 弹性取较小值（保证不弹）
        px_api.CreateFrictionCombineModeAttr().Set("max")
        px_api.CreateRestitutionCombineModeAttr().Set("min")
        binding = UsdShade.MaterialBindingAPI.Apply(root.GetPrim())
        binding.Bind(
            mat,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        print("[factory] baked high-friction physics material (static 10 / dynamic 1.5 / restitution 0.01)")

    stage.Save()
    print(f"[factory] wrote {args_cli.dst}")


main()
simulation_app.close()
