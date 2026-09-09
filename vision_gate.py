"""
M4 连续搬运环节的视觉层
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np

# ---- 闸门相机:仿 D435 的 RGB+深度,分辨率取 1280x720 ----
# (真实 D435 的RGB 流是 1920x1080,所以 1280x720 仍是保守取值。)
GATE_RES = (1280, 720)
GATE_FOCAL = 11.0            # 焦距(mm,Isaac 针孔相机参数)
GATE_APERTURE = 20.955       # 水平光圈/传感器宽度(mm,Isaac 默认标准值)
FX = GATE_RES[0] * GATE_FOCAL / GATE_APERTURE   # 水平焦距(像素)
FY = FX                                          # 方形像素,fy = fx
CX, CY = GATE_RES[0] / 2.0, GATE_RES[1] / 2.0    # 主点 = 图像中心
GATE_PITCH_DEG = 15.0        # 标定结论:2.5 m 站位需要 0~15 度下俯;15 度正好把整排工件放到画面中央
# 头部相机相对骨盆的安装位置(站立时):骨盆 z=0.783,相机 z=1.20
# (head_cam_calib 实测),且在骨盆轴线前方 5 cm
HEAD_CAM_DZ = 0.417          # 相机高出骨盆的距离(m)
HEAD_CAM_DX = 0.05           # 相机在骨盆前方的距离(m)

# 逐类别的"根原点 -> 可见实体中心"z 向偏移(米)。螺母资产的 USD 原点
# 比网格低约 7.5 cm(head_cam_calib 发现);其圆环在 z 缩放 3.75 下高约6 cm,所以可见中心 = 根原点 +0.105 m。
CLS_VIS_Z_OFF = {"cube": 0.0, "bolt": 0.02, "nut": 0.105, "drill": 0.045}
# 逐类别的"视线方向表面 vs 中心"修正量(米)。仅在闸门工作点(2.5 m站位、下俯 15 度,)标定有效
# 换一个视角就不成立:电钻的这个常数在对接位会变号,这也是否决纯视觉对接的原因之一。
CLS_LOS_CORR = {"cube": -0.029, "bolt": -0.039, "nut": -0.030, "drill": -0.054}
# 各类别围绕可见中心的偏航不变，全尺寸(米)——make_yolo_dataset.py
# 用它自动生成标注框,闸门用它推算 ROI 半径
CLS_DIMS = {
    "cube": (0.06, 0.06, 0.06),
    "bolt": (0.16, 0.16, 0.06),
    "nut": (0.23, 0.23, 0.06),
    "drill": (0.24, 0.24, 0.19),
}
CLS_NAMES = ("cube", "bolt", "nut", "drill")   # YOLO 类别 id 0..3 的顺序


def make_gate_camera_cfg(update_period: float):
    #构造可移动头部视角相机的 CameraCfg(在环境配置的__post_init__里调用;渲染器需要 --enable_cameras才会真正生成传感器)。
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    return CameraCfg(
        prim_path="/World/envs/env_.*/VisionGateCam",
        update_period=update_period,
        width=GATE_RES[0], height=GATE_RES[1],
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=GATE_FOCAL, horizontal_aperture=GATE_APERTURE,
            clipping_range=(0.05, 30.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.5), convention="world"),
    )


def cam_quat_world(yaw_rad: float, pitch_down_deg: float):
    #世界约定(x 朝前 / z 朝上)下的相机四元数 (w,x,y,z):先绕 z 偏航 yaw_rad,再向下俯仰 pitch_down_deg 度。
    hy, hp = yaw_rad / 2.0, math.radians(pitch_down_deg) / 2.0
    cy, sy, cp, sp = math.cos(hy), math.sin(hy), math.cos(hp), math.sin(hp)
    return (cy * cp, -sy * sp, cy * sp, sy * cp)


def ros_rot(yaw_rad: float, pitch_down_deg: float) -> np.ndarray:
    #世界系 -> ROS 光学系的旋转矩阵:三列依次是(图像右方向、图像下方向、光轴方向)在世界坐标下的表示。
    pd = math.radians(pitch_down_deg)
    fwd = np.array([math.cos(yaw_rad) * math.cos(pd),
                    math.sin(yaw_rad) * math.cos(pd), -math.sin(pd)])
    left = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0])
    up = np.cross(fwd, left)
    up /= max(np.linalg.norm(up), 1e-9)
    return np.stack([-left, -up, fwd], axis=1)


def head_cam_pose(base_xy, base_z: float, base_yaw: float,
                  pitch_deg: float = GATE_PITCH_DEG):
    #由骨盆的实时状态推算头部相机位姿(等价于真机上跑一遍正运动学)。返回 (t_local[3] 平移, quat_world 四元数, R_ros 旋转矩阵)。
    t = np.array([base_xy[0] + HEAD_CAM_DX * math.cos(base_yaw),
                  base_xy[1] + HEAD_CAM_DX * math.sin(base_yaw),
                  base_z + HEAD_CAM_DZ])
    return t, cam_quat_world(base_yaw, pitch_deg), ros_rot(base_yaw, pitch_deg)


def project(p_local: np.ndarray, t: np.ndarray, R: np.ndarray):
    #世界点 -> (u, v, 相机系 z);点在相机后方(z 过小)时返回 None。
    p_c = R.T @ (p_local - t)
    if p_c[2] < 0.05:
        return None
    return (FX * p_c[0] / p_c[2] + CX, FY * p_c[1] / p_c[2] + CY, p_c[2])


def backproject(u: float, v: float, d: float, t: np.ndarray, R: np.ndarray):
    #像素坐标 + 像平面深度 -> 世界点(project 的逆运算)。
    p_c = np.array([(u - CX) * d / FX, (v - CY) * d / FY, d])
    return R @ p_c + t


def depth_at(depth: np.ndarray, u: float, v: float, half: int = 2):
    #取(u, v)周围(2*half+1)^2邻域内有限有效深度的中位数;邻域内没有有效值时返回 None。用中位数是为了抗单像素噪声/空洞。
    H, W = depth.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        return None
    patch = depth[max(0, vi - half):vi + half + 1,
                  max(0, ui - half):ui + half + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else None


class VisionShadow:
    #--vision_shadow:逐工件记录"纯视觉估计 vs 真值",写入CSV。
    #估计链完全复刻真实部署系统会跑的流程:工件可见中心的像素坐标
    #(用真值投影得到——相当于一个完美检测器;真实 YOLO 只会在此基础上再叠加误差)、深度中位数、反投影、逐类别的视线修正 + 根原点
    #偏移修正。误差在【对接坐标系】里报告(fwd = 机器人朝向,left = 机器人左方)——BC 策略 ±3 cm 容差就是写在这两根轴上的。

    HEADER = ("t_sim,iter,job,entity,cls,is_target,u,v,depth_m,"
              "est_x,est_y,est_z,truth_x,truth_y,truth_z,"
              "err_fwd_cm,err_left_cm,err_z_cm,err_xy_cm\n")

    def __init__(self, csv_path: str):
        self.path = csv_path
        self.n_rows = 0
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(self.HEADER)
        print(f"[vision] shadow log -> {csv_path}")

    def log(self, t_sim: float, it: int, job_idx: int, target_ent: str,
            pieces, cam_t: np.ndarray, cam_R: np.ndarray,
            depth: np.ndarray, base_yaw: float):
        #pieces:可迭代的 (实体名, 类别, 根原点局部坐标 ndarray[3])。
        #返回目标工件的 (err_fwd_cm, err_left_cm),测不到时返回 None。
        # 对接坐标系的两根轴(世界系下的单位向量)
        fwd = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])
        left = np.array([-math.sin(base_yaw), math.cos(base_yaw), 0.0])
        tgt_err = None
        rows = []
        for ent, cls, root in pieces:
            # 根原点 -> 可见实体中心(逐类别 z 偏移)
            vis_c = root + np.array([0.0, 0.0, CLS_VIS_Z_OFF[cls]])
            uvz = project(vis_c, cam_t, cam_R)
            if uvz is None:
                continue                  # 在相机后方/画面外
            u, v, _ = uvz
            d = depth_at(depth, u, v)
            if d is None:
                continue                  # 该处深度无效
            # 反投影得到的是"表面上的可见中心"估计
            est_vis = backproject(u, v, d, cam_t, cam_R)
            los = est_vis - cam_t
            los /= max(np.linalg.norm(los), 1e-9)
            # 两步修正:表面 -> 中心(标定的视线修正),再中心 -> 根原点
            est_root = (est_vis - CLS_LOS_CORR[cls] * los
                        - np.array([0.0, 0.0, CLS_VIS_Z_OFF[cls]]))
            err = est_root - root
            e_fwd, e_left = float(err @ fwd), float(err @ left)
            e_xy = math.hypot(e_fwd, e_left)
            rows.append((round(t_sim, 2), it, job_idx + 1, ent, cls,
                         int(ent == target_ent), round(u, 1), round(v, 1),
                         round(d, 4),
                         round(est_root[0], 4), round(est_root[1], 4),
                         round(est_root[2], 4),
                         round(root[0], 4), round(root[1], 4), round(root[2], 4),
                         round(e_fwd * 100, 2), round(e_left * 100, 2),
                         round(float(err[2]) * 100, 2), round(e_xy * 100, 2)))
            if ent == target_ent:
                tgt_err = (e_fwd * 100, e_left * 100)
        # 追加写并立即关闭:进程崩溃也不会丢已写入的行
        with open(self.path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows(rows)
        self.n_rows += len(rows)
        return tgt_err


class YoloGate:
    #--yolo_gate:对下一个计划工件做"类别身份核对"。

    #check() 在期望像素位置(排产计划工位的投影点——真实部署中这个位置来自厂区地图)周围的ROI圆内找检测框,把检出的类别与计划类别比对。
    #定位信息永远不会取自检测结果,只做"是不是这个东西"的判断。
    

    def __init__(self, model_path: str, conf: float = 0.40):
        from ultralytics import YOLO      # 惰性导入:只有开闸门才加载
        self.model = YOLO(model_path)
        self.conf = conf
        names = self.model.names
        # 模型类别 id -> 我们的类别名(data.yaml 的顺序 = CLS_NAMES)
        self.id2cls = {i: str(n) for i, n in
                       (names.items() if isinstance(names, dict)
                        else enumerate(names))}
        print(f"[vision] YOLO gate loaded: {model_path} classes={self.id2cls}")

    # ROI半宽(米,围绕期望位置)。A桌相邻工位间距的一半是 0.13 m。
    ROI_HALF_M = 0.13

    def detect(self, rgb: np.ndarray) -> list:
        #给实时浮窗叠加层用的纯检测(仅显示用途):返回每个框的(类别名,置信度,(x1, y1, x2, y2))。不做 ROI筛选、不下判定。
        res = self.model.predict(rgb[..., :3], conf=self.conf, verbose=False)[0]
        return [(self.id2cls.get(int(b.cls[0]), "?"), float(b.conf[0]),
                 tuple(float(v) for v in b.xyxy[0]))
                for b in res.boxes]

    def check(self, rgb: np.ndarray, expected_cls: str, expected_uv,
              expected_z: float) -> dict:
        # ROI半径按深度换算成像素:距离越近半径越大;下限 25 px
        radius_px = max(25.0, self.ROI_HALF_M * FX / max(expected_z, 0.3))
        res = self.model.predict(rgb[..., :3], conf=self.conf, verbose=False)[0]
        best = None                       # 取【离期望点最近】的检测,而非置信度最高的
        n_roi = 0
        dets = []                         # 所有检测框,给浮窗显示用
        for b in res.boxes:
            cx, cy = float(b.xywh[0][0]), float(b.xywh[0][1])
            name = self.id2cls.get(int(b.cls[0]), "?")
            conf = float(b.conf[0])
            dets.append((name, conf, tuple(float(v) for v in b.xyxy[0])))
            r = math.hypot(cx - expected_uv[0], cy - expected_uv[1])
            if r > radius_px:
                continue                  # 在 ROI圆外,不参与判定
            n_roi += 1
            if best is None or r < best[2]:
                best = (name, conf, r)
        base = {"dets": dets, "radius_px": radius_px}
        if best is None:
            return {**base, "ok": False, "cls": None, "conf": 0.0,
                    "reason": f"no detection within {radius_px:.0f}px of the "
                              f"expected spot ({len(res.boxes)} elsewhere)"}
        ok = best[0] == expected_cls
        return {**base, "ok": ok, "cls": best[0], "conf": best[1],
                "reason": ("match" if ok else
                           f"saw '{best[0]}' ({best[1]:.2f}) where "
                           f"'{expected_cls}' was scheduled ({n_roi} in ROI)")}


class GateViewer:
    #--gate_view:两个由头部相机供图的悬浮实时窗口——
    #RGB 窗叠加最新的 YOLO 闸门信息(检测框、期望 ROI 圆、PASS/watch状态条),
    #深度窗显示JET伪彩色深度图(近 = 红,远 = 蓝,无效 = 黑)纯显示用途:控制回路绝不读取这里的任何数据。

    RGB_WIN = "G1 head camera - RGB / YOLO gate"
    DEPTH_WIN = "G1 head camera - depth"
    DEPTH_RANGE_M = (0.3, 6.0)            # 深度伪彩色的量程(米)
    UI_RES = (640, 360)                   # omni.ui后端的图像推送分辨率

    def __init__(self, scale: float = 0.5):
        import cv2                        # 惰性导入:只有开浮窗才加载
        self.cv2 = cv2
        self.gate = None                  # 最新叠加层内容,直到被替换才失效
        try:
            w, h = int(GATE_RES[0] * scale), int(GATE_RES[1] * scale)
            for win in (self.RGB_WIN, self.DEPTH_WIN):
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win, w, h)
            self.backend = "cv2"
            self.fast = True
        except cv2.error:                 # headless 版 cv2 -> 改用 omni.ui
            self._init_ui()
            self.backend = "omni.ui"
            # set_data_array 直接推送 numpy 缓冲(跑 30 fps 足够快);
            # 逐元素转 list 的兜底路径慢约 10 倍,只有那条路可走时,调用方会自动把刷新率降下来。
            self.fast = hasattr(self._provs["rgb"], "set_data_array")
        print(f"[vision] gate viewer ({self.backend}): floating RGB+YOLO "
              f"and depth windows")

    def _init_ui(self):
        #创建两个 omni.ui 悬浮窗,各挂一个字节图像提供器。
        import omni.ui as ui
        self._provs, self._wins = {}, {}
        for key, title in (("rgb", self.RGB_WIN), ("depth", self.DEPTH_WIN)):
            prov = ui.ByteImageProvider()
            win = ui.Window(title, width=self.UI_RES[0] + 16,
                            height=self.UI_RES[1] + 40)
            with win.frame:
                ui.ImageWithProvider(
                    prov,
                    fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
            self._provs[key], self._wins[key] = prov, win

    def _push_ui(self, key: str, bgr: np.ndarray):
        #把一帧 BGR 图缩放到 UI_RES、转 RGBA 后推给 omni.ui 窗口。
        cv2 = self.cv2
        small = cv2.resize(bgr, self.UI_RES, interpolation=cv2.INTER_AREA)
        rgba = np.ascontiguousarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGBA))
        prov = self._provs[key]
        w, h = rgba.shape[1], rgba.shape[0]
        try:                              # 快速路径(新版 kit 才有)
            prov.set_data_array(rgba, [w, h])
        except (AttributeError, TypeError):
            prov.set_bytes_data(rgba.flatten().tolist(), [w, h])

    def set_gate(self, res: dict, expected_uv, status: str):
        #记录最新一次闸门判定,供叠加显示(res 来自 YoloGate.check)。判定通过 -> 状态条绿色,否则红色。
        self.gate = (res.get("dets", []), expected_uv,
                     res.get("radius_px", 0.0), status,
                     (60, 200, 60) if res.get("ok") else (50, 60, 230))

    def set_live(self, dets: list):
        #闸门检查间隙的持续显示专用 YOLO 框(没有期望 ROI 圆,状态条为中性黄色)。
        self.gate = (dets, None, 0.0, f"YOLO live: {len(dets)} det",
                     (200, 200, 60))

    def show(self, rgb: np.ndarray, depth: np.ndarray):
        #刷新两个窗口:RGB 帧画上叠加层,深度帧转 JET 伪彩色。
        cv2 = self.cv2
        img = np.ascontiguousarray(rgb[..., :3][..., ::-1])   # RGB -> BGR
        if self.gate is not None:
            dets, uv, rad, status, col = self.gate
            # 所有检测框:绿色矩形 + 类别/置信度标签
            for name, conf, (x1, y1, x2, y2) in dets:
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (60, 220, 60), 2)
                cv2.putText(img, f"{name} {conf:.2f}", (int(x1), int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 220, 60), 2)
            # 期望位置的 ROI 圆(橙色,仅闸门判定时有)
            if uv is not None:
                cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))),
                           max(int(rad), 2), (0, 200, 255), 2)
            # 左上角状态条(PASS/watch/live)
            cv2.putText(img, status, (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        # 深度 -> JET 伪彩色:量程内归一化,近处红、远处蓝、无效像素黑
        d = depth.astype(np.float32)
        valid = np.isfinite(d) & (d > 0)
        lo, hi = self.DEPTH_RANGE_M
        dn = np.zeros_like(d)
        dn[valid] = np.clip((d[valid] - lo) / (hi - lo), 0.0, 1.0)
        dm = cv2.applyColorMap(((1.0 - dn) * 255).astype(np.uint8),
                               cv2.COLORMAP_JET)
        dm[~valid] = 0
        if self.backend == "cv2":
            cv2.imshow(self.RGB_WIN, img)
            cv2.imshow(self.DEPTH_WIN, dm)
            cv2.waitKey(1)                # 驱动 OpenCV 窗口事件循环
        else:
            self._push_ui("rgb", img)
            self._push_ui("depth", dm)


def default_shadow_csv() -> str:
    #默认影子日志文件名:m4_shadow_<时间戳>.csv,放在当前工作目录。
    return os.path.abspath(
        f"m4_shadow_{time.strftime('%Y%m%d_%H%M%S')}.csv")
