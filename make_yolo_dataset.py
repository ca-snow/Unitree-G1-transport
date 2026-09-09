"""
M4 类别闸门的YOLO数据集自动标注生成器
"""

import argparse
import math
import random
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="YOLO gate dataset generator")
parser.add_argument("--out_dir", type=str, default="yolo_gate_data")
parser.add_argument("--num", type=int, default=1200, help="总样本数。")
parser.add_argument("--val_every", type=int, default=10,
                    help="每第 N 个样本划入验证集(val split)。")
parser.add_argument("--settle_steps", type=int, default=25,
                    help="每次工件抖动后跑多少步物理让其沉降。")
parser.add_argument("--warmup_frames", type=int, default=4,
                    help="每次移动相机后渲染多少帧(让渲染管线出新图)。")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True          # 必须启用相机渲染管线
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

# 搬运场景配置放在 unitree_sim_isaaclab 工程下,把工程根加进 sys.path
PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from tasks.rl.g1_transport.transport_scene_cfg import TransportSceneCfg

# 同目录导入(本文件与 vision_gate.py 一起放在 scripts 目录里)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision_gate import (
    CLS_DIMS, CLS_NAMES, CLS_VIS_Z_OFF, GATE_RES,
    cam_quat_world, depth_at, make_gate_camera_cfg, project, ros_rot,
)

# 场景里的 7 个工件:(场景实体名, YOLO 类别名)
PIECES = (
    ("object", "cube"), ("bolt", "bolt"), ("object2", "cube"),
    ("drill", "drill"), ("bolt2", "bolt"), ("nut", "nut"), ("drill2", "drill"),
)
W, H = GATE_RES


def randomize_lights(rng: random.Random):
    #域随机化:把场景里每盏灯的强度缩放到原值的0.3~2.0倍(原始强度在第一次调用时缓存下来,之后每次都从原值出发缩放)。
    try:
        import omni.usd
        from pxr import UsdLux
    except ImportError:
        return
    stage = omni.usd.get_context().get_stage()
    if not hasattr(randomize_lights, "_base"):
        randomize_lights._base = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.LightAPI) or prim.GetTypeName().endswith("Light"):
                attr = prim.GetAttribute("inputs:intensity")
                if attr and attr.Get() is not None:
                    randomize_lights._base[str(prim.GetPath())] = float(attr.Get())
    for path, base in randomize_lights._base.items():
        prim = stage.GetPrimAtPath(path)
        if prim:
            prim.GetAttribute("inputs:intensity").Set(
                base * rng.uniform(0.3, 2.0))


def main():
    rng = random.Random(args_cli.seed)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    scene_cfg = TransportSceneCfg(num_envs=1, env_spacing=20.0)
    scene_cfg.gate_cam = make_gate_camera_cfg(0.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    dt = sim.get_physics_dt()
    #先空跑60步物理,让所有工件在桌面上完全沉降
    for _ in range(60):
        sim.step(render=False)
    scene.update(dt)
    cam = scene["gate_cam"]
    origin = scene.env_origins[0].cpu().numpy()
    device = scene.device

    # 记录沉降后的标称位姿——后续每帧的抖动都以此为基准
    base_state = {ent: scene[ent].data.root_state_w[0].clone()
                  for ent, _ in PIECES}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args_cli.out_dir, sub), exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        print("[dataset] FATAL: PIL required to write images")
        return
    # 写 ultralytics 的 data.yaml(类别顺序 = CLS_NAMES = 模型 id 顺序)
    with open(os.path.join(args_cli.out_dir, "data.yaml"), "w",
              encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args_cli.out_dir)}\n"
                "train: images/train\nval: images/val\n"
                f"names: {list(CLS_NAMES)}\n")

    n_written = 0
    n_boxes = 0
    for i in range(args_cli.num):
        # 1. 每个工件围绕标称位姿重新抖动
        for ent, _ in PIECES:
            st = base_state[ent].clone().unsqueeze(0)
            st[0, 0] += rng.uniform(-0.03, 0.03)
            st[0, 1] += rng.uniform(-0.03, 0.03)
            st[0, 2] += 0.002                      # 微微抬高一点再落下,沉降干净
            half = rng.uniform(-math.pi, math.pi) / 2.0
            # 在标称姿态之上再叠加一个随机 z 轴偏航(四元数乘法展开)
            qw0, qx0, qy0, qz0 = st[0, 3:7].tolist()
            cw, sz = math.cos(half), math.sin(half)
            st[0, 3] = cw * qw0 - sz * qz0
            st[0, 4] = cw * qx0 - sz * qy0
            st[0, 5] = cw * qy0 + sz * qx0
            st[0, 6] = cw * qz0 + sz * qw0
            st[0, 7:] = 0.0                        # 速度清零
            scene[ent].write_root_state_to_sim(st)
        for _ in range(args_cli.settle_steps):
            sim.step(render=False)
        scene.update(dt)

        # 2. 随机化的闸门视角,瞄准轮换选中的某一个工位
        tgt_ent, _ = PIECES[i % len(PIECES)]
        p = scene[tgt_ent].data.root_pos_w[0].cpu().numpy() - origin
        stand = rng.uniform(1.6, 3.2)              # 站位距离(m)
        cam_t = np.array([p[0] + stand,
                          p[1] + rng.uniform(-0.4, 0.4),
                          rng.uniform(1.05, 1.35)])
        yaw = math.pi + math.radians(rng.uniform(-10.0, 10.0))
        pitch = rng.uniform(3.0, 28.0)
        cam.set_world_poses(
            torch.tensor([(origin + cam_t).tolist()], dtype=torch.float32,device=device),
            torch.tensor([cam_quat_world(yaw, pitch)], dtype=torch.float32,device=device),
            convention="world")
        randomize_lights(rng)
        # 移动相机后要渲染几帧,等RTX管线把新视角的图像刷出来
        for _ in range(args_cli.warmup_frames):
            sim.step(render=True)
        scene.update(dt)

        rgb = cam.data.output["rgb"][0].cpu().numpy()
        depth = cam.data.output["distance_to_image_plane"][0].cpu().numpy()
        depth = depth.reshape(H, W) if depth.ndim > 2 else depth
        R = ros_rot(yaw, pitch)

        # 3. 自动标注:投影每个工件的 3D 包围盒,并做遮挡检查
        labels = []
        for ent, cls in PIECES:
            root = scene[ent].data.root_pos_w[0].cpu().numpy() - origin
            # 根原点 -> 可见实体中心(逐类别 z 偏移)
            c = root + np.array([0.0, 0.0, CLS_VIS_Z_OFF[cls]])
            uvz = project(c, cam_t, R)
            if uvz is None or not (0 <= uvz[0] < W and 0 <= uvz[1] < H):
                continue                            # 中心不在画面内,跳过
            # 把 3D 包围盒的 8 个角点全部投影,取像素外接矩形
            dx, dy, dz = CLS_DIMS[cls]
            us, vs = [], []
            for sx in (-0.5, 0.5):
                for sy in (-0.5, 0.5):
                    for sz_ in (-0.5, 0.5):
                        q = c + np.array([sx * dx, sy * dy, sz_ * dz])
                        r = project(q, cam_t, R)
                        if r is not None:
                            us.append(r[0])
                            vs.append(r[1])
            if len(us) < 8:
                continue                            # 有角点投影失败,弃用
            u0, u1 = max(0.0, min(us)), min(float(W), max(us))
            v0, v1 = max(0.0, min(vs)), min(float(H), max(vs))
            if u1 - u0 < 6 or v1 - v0 < 4:
                continue                            # 框太小,学不到东西
            # 遮挡检查:在框内取 3x3 网格采样深度任一采样点的实测深度与期望深度差 <0.25 m即认为该工件可见。
            visible = False
            for fu in (0.3, 0.5, 0.7):
                for fv in (0.3, 0.5, 0.7):
                    d_meas = depth_at(depth, u0 + fu * (u1 - u0),
                                      v0 + fv * (v1 - v0), half=1)
                    if d_meas is not None and abs(d_meas - uvz[2]) < 0.25:
                        visible = True
                        break
                if visible:
                    break
            if not visible:
                continue
            # YOLO 格式:类别 id、框中心 xy、框宽高(全部归一化到 0~1)
            labels.append((CLS_NAMES.index(cls),
                           (u0 + u1) / 2.0 / W, (v0 + v1) / 2.0 / H,
                           (u1 - u0) / W, (v1 - v0) / H))
        if not labels:
            continue                                # 整帧没有可用标注,丢弃

        split = "val" if (i % args_cli.val_every) == 0 else "train"
        stem = f"gate_{i:05d}"
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
            os.path.join(args_cli.out_dir, "images", split, stem + ".jpg"),
            quality=92)
        with open(os.path.join(args_cli.out_dir, "labels", split,
                               stem + ".txt"), "w", encoding="utf-8") as f:
            for cid, xc, yc, bw, bh in labels:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        n_written += 1
        n_boxes += len(labels)
        if n_written % 100 == 0:
            print(f"[dataset] {n_written}/{args_cli.num} images "
                  f"({n_boxes} boxes so far)")

    print(f"[dataset] DONE: {n_written} images, {n_boxes} boxes -> "
          f"{os.path.abspath(args_cli.out_dir)}")
    print("[dataset] train: yolo detect train "
          f"data={os.path.abspath(args_cli.out_dir)}/data.yaml "
          "model=yolov8n.pt imgsz=1280 epochs=60 batch=16 name=gate_v1")


if __name__ == "__main__":
    main()
    print("[dataset] closing app...")
    try:
        simulation_app.close()
    finally:
        os._exit(0)
