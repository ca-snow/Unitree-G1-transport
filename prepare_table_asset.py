# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
从 table_with_yellowbox.usd 派生出一张"干净"的打包桌（去掉桌面上的黄色盒子）。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Strip prims from the packing-table USD.")
# --src：带黄盒子的源桌子 USD（项目 assets 目录里的文件）
parser.add_argument(
    "--src", type=str,
    default="/home/0000_wy/unitree/unitree_sim_isaaclab/assets/objects/table_with_yellowbox.usd",
    help="Source table USD.")
# --dst：输出的干净桌子 USD 路径（已存在则覆盖）
parser.add_argument(
    "--dst", type=str,
    default="/home/0000_wy/unitree/unitree_sim_isaaclab/assets/objects/table_plain.usd",
    help="Output USD path (overwritten).")
# --deactivate：要停用的源 prim 路径（逗号分隔，就是第 1 步检查时打印出来的路径）。
# 不给此参数则只打印 prim 树（即第 1 步检查模式）
parser.add_argument(
    "--deactivate", type=str, default=None,
    help="Comma-separated SOURCE prim paths (as printed by the inspect run) to deactivate. "
    "If omitted, the script only prints the prim tree.")
# --max_depth：打印 prim 树的最大深度。默认 4 层——足够看到桌子的部件级节点，
# 又不会被叶子网格刷屏
parser.add_argument("--max_depth", type=int, default=4, help="Tree print depth.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# 启动 Isaac Sim（USD 库随它提供）
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from pxr import Usd, UsdGeom


def print_tree(stage: Usd.Stage, max_depth: int):
    #打印限深的 prim 树（第 1 步检查模式用），帮人工定位黄盒子的路径。

    #每行输出：路径、类型、该节点之下的网格数量（meshes_below）——
    #黄盒子通常是桌面网格的兄弟节点，名字里多半带 'crate' 或 'box'。
    default_prim = stage.GetDefaultPrim()
    print(f"[table] defaultPrim = {default_prim.GetPath()}")
    print("[table] prim tree (depth-limited); the yellow box is likely a 'crate'/'box'-named sibling of the table mesh:")
    for prim in stage.Traverse():
        # 用路径里 "/" 的个数当作深度，超过 max_depth 的节点跳过不打印
        depth = str(prim.GetPath()).count("/")
        if depth > max_depth:
            continue
        kind = prim.GetTypeName()
        # 统计该节点子树里有多少个网格：网格数为 0 的分支多半只是变换节点
        n_mesh = sum(1 for c in Usd.PrimRange(prim) if c.IsA(UsdGeom.Mesh))
        indent = "  " * depth
        print(f"[table] {indent}{prim.GetPath()}  <{kind}>  meshes_below={n_mesh}")


def main():
    #无 --deactivate 时打印 prim 树；有则停用指定 prim 并导出扁平化的干净 USD。
    src_stage = Usd.Stage.Open(args_cli.src)
    if src_stage is None:
        print(f"[table] ERROR: cannot open {args_cli.src}")
        return

    # 第 1 步：检查模式——只打印树，提示下一步怎么跑，然后结束
    if not args_cli.deactivate:
        print_tree(src_stage, args_cli.max_depth)
        print("[table] Re-run with --deactivate <path[,path]> to write the clean table.")
        return

    default_prim = src_stage.GetDefaultPrim()
    src_default = str(default_prim.GetPath())

    # 第 2 步：构建模式。在"内存中"直接停用打开的源 stage 上的 prim
    # （源文件从头到尾不执行保存，绝不会被改坏），然后用 Flatten（扁平化）
    # 导出：结果确定、不牵扯引用解析。路径写法宽容：带不带 defaultPrim
    # 前缀都接受（例如 /PackingTable_2/Cube 和 /Root/PackingTable_2/Cube ）。
    n_off = 0  # 成功停用的 prim 计数
    for raw in [p.strip() for p in args_cli.deactivate.split(",") if p.strip()]:
        prim = src_stage.GetPrimAtPath(raw)
        if not prim:
            # 原样找不到就试着补上 defaultPrim 前缀再找一次
            prim = src_stage.GetPrimAtPath(src_default + raw)
        if not prim:
            print(f"[table] ERROR: neither {raw} nor {src_default + raw} exists; skipped "
                  f"(run without --deactivate to print the tree)")
            continue
        prim.SetActive(False)  # 停用：该 prim（含几何和碰撞体）从合成结果中剔除
        n_off += 1
        print(f"[table] deactivated {prim.GetPath()}")

    # 一个都没停用成功就不写输出文件，避免生成一张"没去掉盒子的假干净桌"
    if n_off == 0:
        print("[table] ERROR: nothing deactivated - NOT writing an output file.")
        return

    os.makedirs(os.path.dirname(args_cli.dst), exist_ok=True)
    if os.path.exists(args_cli.dst):
        os.remove(args_cli.dst)
    # Flatten：把（已带停用覆盖的）合成结果压成一个自包含的层再导出
    layer = src_stage.Flatten()
    layer.defaultPrim = default_prim.GetName()  # 保持默认 prim 名不变，供场景配置引用
    layer.Export(args_cli.dst)
    print(f"[table] wrote {args_cli.dst} (flattened, source file untouched)")


main()
simulation_app.close()
