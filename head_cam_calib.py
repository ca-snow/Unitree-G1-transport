"""
G1 头部相机标定 / 偏差测量装置(视觉阶段第 1 步)。

在做任何 YOLO / 闭环工作之前,先定量回答三个问题:
  1. 视场覆盖:头部相机在哪些站位距离 / 俯仰角下能看到A桌的每个
     抓取工位(对接位时工件在身前约 0.4 m、身下约 0.4 m 处)。
  2. 投影链误差:真值根坐标 -> 像素 -> 深度 -> 反投影回3D;残差在
     垂直于视线方向上的分量,就是纯视觉对接目标会继承的几何误差。
  3. 逐类别"表面 vs 根原点"偏移:误差沿视线方向的分量(深度图测到的
     是朝向相机的表面,不是资产的 USD 根原点)。
"""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 head-camera calibration rig")
parser.add_argument("--out_dir", type=str, default="head_cam_calib_out")
parser.add_argument("--pitches", type=str, default="0,15,30,45,60",
                    help="每个测量位姿都会扫过的相机下俯角列表(度)。")
parser.add_argument("--cam_z", type=float, default=1.20,
                    help="头部相机离地高度(m)。")
parser.add_argument("--cam_dx", type=float, default=0.05,
                    help="相机相对骨盆轴线的前向偏移(m)。")
parser.add_argument("--stagings", type=str, default="2.5,1.5,0.8",
                    help="除对接位之外要测的站位距离列表(m)。")
parser.add_argument("--settle_steps", type=int, default=60)
parser.add_argument("--warmup_frames", type=int, default=8,
                    help="每次移动相机后的渲染+更新步数。")
parser.add_argument("--debug", action="store_true",
                    help="在第一个测量点打印全部中间量(传感器位姿回读 vs 指令值、内参 K、深度图统计)。")
parser.add_argument("--check_only", action="store_true",
                    help="功能冒烟测试(约 1 分钟):只采集一个位姿"
                    "(第一个对接点、俯仰 45 度),打印 RGB/深度的 "
                    "PASS/FAIL 判定,保存 check_rgb.png + check_depth.png "
                    "后退出。请【先】跑这个确认深度标注器工作正常,"
                    "再跑完整扫描。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.check_only:
    args_cli.debug = True
args_cli.enable_cameras = True          # RTX 传感器必须在应用启动前声明
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import os

PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import CameraCfg

from tasks.rl.g1_transport.transport_scene_cfg import (
    BC_SPAWN_FWD,
    BC_SPAWN_RIGHT,
    PICK_DOCK_AHEAD,
    PICK_DOCK_RIGHT,
    TransportSceneCfg,
)

# 场景里的 7 个工件:(场景实体名, 类别名)
PIECES = (
    ("object", "cube"), ("bolt", "bolt"), ("object2", "cube"),
    ("drill", "drill"), ("bolt2", "bolt"), ("nut", "nut"), ("drill2", "drill"),
)

# 仿 D435 参数:848x480,水平视场角 87 度 -> 垂直视场角约 57 度。
CAM_RES = (848, 480)
CAM_FOCAL = 11.0
CAM_APERTURE = 20.955
# 解析法针孔内参——【全部计算只用它】fx = 宽度 * 焦距 / 水平光圈;方形像素;主点取图像中心。
FX = CAM_RES[0] * CAM_FOCAL / CAM_APERTURE
FY = FX
CX, CY = CAM_RES[0] / 2.0, CAM_RES[1] / 2.0


def cam_quat_world(yaw_deg: float, pitch_down_deg: float):
    #世界约定(x 朝前 / z 朝上)下的相机四元数 (w,x,y,z)。
    hy, hp = math.radians(yaw_deg) / 2.0, math.radians(pitch_down_deg) / 2.0
    cy, sy, cp, sp = math.cos(hy), math.sin(hy), math.cos(hp), math.sin(hp)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def ros_rot_from_yaw_pitch(yaw_deg: float, pitch_down_deg: float):
    #直接由偏航+俯仰构造 3x3 的世界系->光学系旋转矩阵(v2:不做任何传感器回读)。三列是 ROS 光学系坐标轴在世界坐标下的表示:
    #x_ros = 图像右方向,y_ros = 图像下方向,z_ros = 光轴方向。
    yw, pd = math.radians(yaw_deg), math.radians(pitch_down_deg)
    fwd = np.array([math.cos(yw) * math.cos(pd),
                    math.sin(yw) * math.cos(pd), -math.sin(pd)])
    left = np.array([-math.sin(yw), math.cos(yw), 0.0])
    up = np.cross(fwd, left) / max(np.linalg.norm(np.cross(fwd, left)), 1e-9)
    # ROS 光学系:x = 右 = -left,y = 下 = -up,z = 前 = fwd
    return np.stack([-left, -up, fwd], axis=1)


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    scene_cfg = TransportSceneCfg(num_envs=1, env_spacing=20.0)
    # 挂一台标定专用相机(初始位姿随意,后面每个测量点都会重设)
    scene_cfg.head_cam = CameraCfg(
        prim_path="/World/envs/env_.*/HeadCalibCam",
        update_period=0.0,
        width=CAM_RES[0], height=CAM_RES[1],
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=CAM_FOCAL, horizontal_aperture=CAM_APERTURE,
            clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), convention="world"),
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    dt = sim.get_physics_dt()
    # 空跑物理,让工件在桌面上沉降到位
    for _ in range(args_cli.settle_steps):
        sim.step(render=False)
    scene.update(dt)
    cam = scene["head_cam"]
    origin = scene.env_origins[0].cpu().numpy()

    # 记录每个工件沉降后的根坐标(环境局部系)
    piece_pos = {}
    for ent, cls in PIECES:
        piece_pos[ent] = scene[ent].data.root_pos_w[0].cpu().numpy() - origin
        if args_cli.debug:
            print(f"[calib][debug] piece {ent:8s} root (local) = {piece_pos[ent]}")

    # 测量点:每个工件的对接位姿 + 各站位距离(机器人面朝 -x 方向:骨盆 x = 工件 x + ahead,骨盆 y = 工件 y - right)
    points = []
    stagings = [float(s) for s in args_cli.stagings.split(",") if s.strip()]
    for ent, cls in PIECES:
        px, py = float(piece_pos[ent][0]), float(piece_pos[ent][1])
        ahead = PICK_DOCK_AHEAD + BC_SPAWN_FWD[cls]
        right = PICK_DOCK_RIGHT + BC_SPAWN_RIGHT.get(cls, 0.0)
        points.append((f"{ent}@dock", ent, (px + ahead, py - right)))
        for s in stagings:
            points.append((f"{ent}@{s:.1f}m", ent, (px + s, py - right)))

    pitches = [float(p) for p in args_cli.pitches.split(",") if p.strip()]
    if args_cli.check_only:
        points = points[:1]             # 只测第一个对接位姿
        pitches = [45.0]                # 45 度下俯保证工件在画面中央附近
        print("[calib] CHECK-ONLY mode: 1 pose, pitch 45 deg")
    os.makedirs(args_cli.out_dir, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
        have_pil = True
    except ImportError:
        have_pil = False
        print("[calib] PIL missing - annotated frames saved as .npy instead")

    W, H = CAM_RES
    rows = []
    device = scene.device
    first_debug = args_cli.debug
    n_png = 0
    for name, target_ent, (gx, gy) in points:
        # 相机位置 = 骨盆位置往后退 cam_dx(因为面朝 -x,相机在骨盆"前方"即 -x 方向,这里以骨盆坐标扣除偏移),高度取 cam_z
        cam_pos_local = np.array([gx - args_cli.cam_dx, gy, args_cli.cam_z])
        for pitch in pitches:
            pos_t = torch.tensor([(origin + cam_pos_local).tolist()],
                                 dtype=torch.float32, device=device)
            quat_t = torch.tensor([cam_quat_world(180.0, pitch)],
                                  dtype=torch.float32, device=device)
            cam.set_world_poses(pos_t, quat_t, convention="world")
            # 渲染几帧让 RTX 管线出新视角的图
            for _ in range(args_cli.warmup_frames):
                sim.step(render=True)
                scene.update(dt)

            rgb = cam.data.output["rgb"][0].cpu().numpy()
            depth = cam.data.output["distance_to_image_plane"][0].cpu().numpy()
            depth = depth.reshape(H, W) if depth.ndim > 2 else depth
            # v2:位姿/内参全部用确定性的指令值,不读传感器缓冲
            t = cam_pos_local
            Rr = ros_rot_from_yaw_pitch(180.0, pitch)

            if first_debug:
                first_debug = False
                fin = depth[np.isfinite(depth)]
                print(f"[calib][debug] commanded cam pos (local) = {t}, "
                      f"pitch {pitch} deg")
                print(f"[calib][debug] sensor readback pos_w = "
                      f"{cam.data.pos_w[0].cpu().numpy() - origin} "
                      f"quat_w_ros = {cam.data.quat_w_ros[0].cpu().numpy()}")
                print(f"[calib][debug] sensor intrinsics =\n"
                      f"{cam.data.intrinsic_matrices[0].cpu().numpy()}")
                print(f"[calib][debug] analytic K: fx={FX:.1f} fy={FY:.1f} "
                      f"cx={CX:.1f} cy={CY:.1f}")
                print(f"[calib][debug] rgb shape={rgb.shape} dtype={rgb.dtype} "
                      f"mean={rgb[..., :3].mean():.1f}")
                print(f"[calib][debug] depth finite px={fin.size}/{depth.size} "
                      + (f"range=[{fin.min():.3f},{fin.max():.3f}]" if fin.size else ""))
                p_dbg = piece_pos[target_ent]
                p_c_dbg = Rr.T @ (p_dbg - t)
                print(f"[calib][debug] target {target_ent} p_cam(ros) = {p_c_dbg}")

                if args_cli.check_only:
                    # ---- 功能冒烟测试:打判定 + 存原始帧 ----
                    ok_rgb = rgb[..., :3].mean() > 2.0
                    frac = fin.size / max(depth.size, 1)
                    # 相机从约 0.6 m 高度俯视桌面:有效深度应覆盖大部分像素,且中位数落在合理范围内
                    ok_depth = (frac > 0.5 and fin.size > 0
                                and 0.1 < float(np.median(fin)) < 5.0)
                    u_dbg = FX * p_c_dbg[0] / p_c_dbg[2] + CX
                    v_dbg = FY * p_c_dbg[1] / p_c_dbg[2] + CY
                    d_pix = float("nan")
                    if 0 <= u_dbg < W and 0 <= v_dbg < H and fin.size:
                        d_pix = float(depth[int(v_dbg), int(u_dbg)])
                    d_true = float(np.linalg.norm(p_dbg - t))
                    rgb_verdict = "PASS" if ok_rgb else "FAIL (image is black)"
                    depth_verdict = ("PASS" if ok_depth else
                                     "FAIL (annotator returned no/degenerate "
                                     "data - depth pipeline is OFF)")
                    print(f"[calib][check] RGB   : {rgb_verdict}")
                    print(f"[calib][check] DEPTH : {depth_verdict} "
                          f"(finite {frac * 100:.0f}% of pixels)")
                    print(f"[calib][check] target pixel ({u_dbg:.0f},{v_dbg:.0f}) "
                          f"depth={d_pix:.3f} m vs truth range {d_true:.3f} m "
                          f"(should differ only by the piece surface offset, "
                          f"i.e. < ~5 cm)")
                    if have_pil:
                        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
                            os.path.join(args_cli.out_dir, "check_rgb.png"))
                        dvis = np.nan_to_num(depth, nan=0.0, posinf=0.0,
                                             neginf=0.0)
                        dvis = (np.clip(dvis, 0.0, 3.0) / 3.0 * 255).astype(
                            np.uint8)
                        Image.fromarray(dvis).save(
                            os.path.join(args_cli.out_dir, "check_depth.png"))
                        print(f"[calib][check] frames -> "
                              f"{args_cli.out_dir}/check_rgb.png, check_depth.png")
                    print("[calib][check] verdict: "
                          + ("ALL PASS - safe to run the full sweep"
                             if ok_rgb and ok_depth else
                             "FAILED - do NOT run the full sweep; send me the "
                             "lines above"))
                    return

            # 对画面里的每个工件跑一遍完整的投影/反投影链,记录误差
            for ent, cls in PIECES:
                p_w = piece_pos[ent]
                p_c = Rr.T @ (p_w - t)
                if not np.isfinite(p_c).all() or p_c[2] < 0.05:
                    continue              # 在相机后方,无法投影
                u = FX * p_c[0] / p_c[2] + CX
                v = FY * p_c[1] / p_c[2] + CY
                in_fov = bool(0 <= u < W and 0 <= v < H)
                d = err_tot = err_los = err_res = float("nan")
                if in_fov:
                    # 5x5 邻域深度中位数(抗噪声/空洞)
                    ui, vi = int(round(u)), int(round(v))
                    patch = depth[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
                    patch = patch[np.isfinite(patch) & (patch > 0)]
                    if patch.size:
                        d = float(np.median(patch))
                        # 反投影回世界系,和真值求误差,再把误差分解为:
                        # 沿视线分量(= 表面 vs 根原点偏移)和垂直视线的残差分量(= 纯几何误差,对接会直接继承它)
                        p_est_c = np.array([(u - CX) * d / FX,
                                            (v - CY) * d / FY, d])
                        p_est_w = Rr @ p_est_c + t
                        err = p_est_w - p_w
                        los = (p_w - t) / np.linalg.norm(p_w - t)
                        err_los = float(np.dot(err, los))
                        err_res = float(np.linalg.norm(err - err_los * los))
                        err_tot = float(np.linalg.norm(err))
                rows.append((name, pitch, ent, cls, int(ent == target_ent),
                             int(in_fov), round(float(u), 1), round(float(v), 1),
                             None if math.isnan(d) else round(d, 4),
                             None if math.isnan(err_tot) else round(err_tot * 100, 2),
                             None if math.isnan(err_los) else round(err_los * 100, 2),
                             None if math.isnan(err_res) else round(err_res * 100, 2)))

            # 对接位和第一个站位距离的帧存成标注图(圈出画面内的工件)
            if name.endswith("@dock") or name.endswith(f"@{stagings[0]:.1f}m"):
                frame = rgb[..., :3].astype(np.uint8).copy()
                tag = f"{name}_p{int(pitch)}".replace("@", "_")
                fpath = os.path.join(args_cli.out_dir, tag + ".png")
                if have_pil:
                    im = Image.fromarray(frame)
                    dr = ImageDraw.Draw(im)
                    for r in rows[-len(PIECES):]:
                        if r[0] != name or r[1] != pitch or r[5] != 1:
                            continue
                        # 目标工件红圈,其余黄圈
                        col = (255, 0, 0) if r[4] else (255, 255, 0)
                        u, v = r[6], r[7]
                        dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=col, width=2)
                        dr.text((u + 8, v - 6), r[2], fill=col)
                    im.save(fpath)
                else:
                    fpath = fpath[:-4] + ".npy"
                    np.save(fpath, frame)
                n_png += 1
    print(f"[calib] wrote {n_png} annotated frames to {args_cli.out_dir}")

    # 完整数据表落盘
    csv_path = os.path.join(args_cli.out_dir, "calib_report.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("point,pitch_deg,entity,cls,is_target,in_fov,u,v,depth_m,"
                "err_total_cm,err_line_of_sight_cm,err_residual_cm\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")
    print(f"[calib] full table -> {csv_path}")

    # 汇总 1:各俯仰角下,目标工件在各距离处的可见率
    print("\n[calib] ===== SUMMARY: target piece visibility per pitch =====")
    dist_names = ["dock"] + [f"{s:.1f}m" for s in stagings]
    for pitch in pitches:
        parts = []
        for dn in dist_names:
            tgt = [r for r in rows if r[1] == pitch and r[4] == 1
                   and r[0].endswith("@" + dn)]
            vis = sum(1 for r in tgt if r[5] == 1 and r[8] is not None)
            parts.append(f"{dn}: {vis}/{len(tgt)}")
        print(f"[calib]   pitch {pitch:4.0f} deg  " + "  ".join(parts))

    # 汇总 2:误差预算(只统计目标可见的样本)
    print("\n[calib] ===== SUMMARY: error budget (target visible only) =====")
    for dn in dist_names:
        sel = [r for r in rows if r[4] == 1 and r[5] == 1 and r[11] is not None
               and r[0].endswith("@" + dn)]
        if not sel:
            print(f"[calib]   {dn:>5}: never visible")
            continue
        res = np.array([r[11] for r in sel])
        los = np.array([r[10] for r in sel])
        print(f"[calib]   {dn:>5}: residual (geometry) err "
              f"{res.mean():.2f}+-{res.std():.2f} cm (max {res.max():.2f}) | "
              f"line-of-sight (surface-root) {los.mean():+.2f}+-{los.std():.2f} cm "
              f"| n={len(sel)}")
    # 汇总 3:逐类别的"表面 vs 根原点"偏移
    print("\n[calib] per-class surface-vs-root offsets:")
    for cls in ("cube", "bolt", "nut", "drill"):
        sel = [r for r in rows if r[3] == cls and r[5] == 1 and r[10] is not None]
        if sel:
            los = np.array([r[10] for r in sel])
            print(f"[calib]   {cls:>5}: {los.mean():+.2f}+-{los.std():.2f} cm "
                  f"(n={len(sel)})")
    print("[calib] done.")


if __name__ == "__main__":
    main()
    print("[calib] closing app...")
    try:
        simulation_app.close()
    finally:
        os._exit(0)
