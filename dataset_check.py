# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
抓放（pick-and-place）专家 .npz 数据集的离线体检工具（不需要 Isaac）。
"""

import argparse

import numpy as np

HORIZON = 725           # grasp-expert.py 每条 rollout 的长度（一个完整抓放周期）
MAX_EPISODE_LEN = 800   # 环境的最大步数：8 秒 / (0.005 秒物理步 * decimation 2)

RIGHT_OBJ_REL = slice(75, 78)       # 物体相对右腕的位置向量
LEFT_OBJ_REL = slice(72, 75)        # 物体相对左腕的位置向量
PHASE_DIM = 106                     # episode 相位所在的观测维
PLACE_TARGET_REL = slice(107, 110)  # 放置目标相对右腕的位置向量
ARM_ACT = slice(0, 14)      # 关节树顺序：前 14 维全是手臂
FINGER_ACT = slice(14, 28)  # 后 14 维全是手指（左右手交错排列）

# 各相位边界的步号 -> 相位名（与 grasp-expert.py 的时序一致），用于打点表
PHASE_CHECKPOINTS = {
    19: "HOLD", 64: "UP", 104: "TRAV", 184: "DESCEND", 244: "INSERT",
    324: "CLOSE", 404: "LIFT", 504: "CARRY", 584: "LOWER",
    634: "RELEASE", 724: "END",
}

CHECK_PROFILES = {
    "cube": dict(preclose_dims=0, insert_prox=0.18, hold_drift=0.05),
    "bolt": dict(preclose_dims=0, insert_prox=0.18, hold_drift=0.05),
    "nut": dict(preclose_dims=4, insert_prox=0.30, hold_drift=0.05),
    "drill": dict(preclose_dims=0, insert_prox=0.30, hold_drift=0.08),
}


def check(name: str, ok: bool, detail: str):
    #打印一行 [OK]/[BAD] 检查结果并原样返回 ok（便于用 &= 聚合总结果）。
    print(f"  [{'OK ' if ok else 'BAD'}] {name}: {detail}")
    return ok


def main():
    #体检入口：逐个 .npz 文件跑维度/相位/抓取特征/手指动作等一整套检查。
    parser = argparse.ArgumentParser(description="Offline pick-and-place dataset health check.")
    # --dataset：待检查的 .npz 路径，可逗号分隔多个（逐个独立检查）
    parser.add_argument("--dataset", type=str, required=True, help="Path(s) to .npz, comma-separated.")
    # --workpiece：数据采集时用的工件（决定 pre-close/接近距离/持握漂移
    # 三项期望值，见 CHECK_PROFILES 注释）。一个数据文件 = 一种工件。
    parser.add_argument(
        "--workpiece", type=str, default="cube", choices=sorted(CHECK_PROFILES),
        help="Workpiece the dataset was collected with (adjusts the pre-close / proximity / "
        "hold-drift expectations, see CHECK_PROFILES). One dataset file = one workpiece.")
    args = parser.parse_args()
    prof = CHECK_PROFILES[args.workpiece]

    all_ok = True   # 所有文件所有检查项的总开关（任一 BAD 即 False）
    for path in args.dataset.split(","):
        path = path.strip()
        try:
            data = np.load(path)
        except FileNotFoundError:
            print(f"\n=== {path} ===")
            all_ok &= check("file exists", False, "not found - check the name with: ls data/*.npz")
            continue
        obs, act = data["obs"], data["actions"]
        print(f"\n=== {path} ===")
        print(f"  obs {obs.shape}  actions {act.shape}")

        # 检查 1：维度必须是 obs 110 / action 28（本管线的固定布局）
        all_ok &= check("dims", obs.shape[1] == 110 and act.shape[1] == 28,
                        f"obs_dim={obs.shape[1]} (want 110), act_dim={act.shape[1]} (want 28)")
        # 检查 2：总行数必须能被 725 整除（即整数条完整 episode）
        n_ep = obs.shape[0] / HORIZON
        all_ok &= check("episodes", n_ep == int(n_ep),
                        f"{obs.shape[0]} rows / {HORIZON} = {n_ep:.2f} episodes")
        n_ep = int(n_ep)

        t = np.arange(obs.shape[0]) // n_ep                                # 由行号推出的真实时间步
        phase_t = np.rint(obs[:, PHASE_DIM] * MAX_EPISODE_LEN).astype(int)  # 观测自己声称的时间步
        # 两者不一致的行属于哪些 episode（行号 % n_ep = episode 编号）
        bad_ep = np.unique((np.arange(obs.shape[0]) % n_ep)[phase_t != t])
        all_ok &= check("no mid-episode resets", len(bad_ep) == 0,
                        f"{len(bad_ep)}/{n_ep} episodes have a phase clock restart "
                        f"(env auto-reset mid-rollout; excluded from checks below)")

        # 剔除脏 episode，然后在干净的 [HORIZON, n_ep, dim] 三维视图上继续检查
        keep = np.setdiff1d(np.arange(n_ep), bad_ep)
        obs3 = obs.reshape(HORIZON, n_ep, -1)[:, keep, :]
        act3 = act.reshape(HORIZON, n_ep, -1)[:, keep, :]
        obs = obs3.reshape(-1, obs.shape[1])
        act = act3.reshape(-1, act.shape[1])
        n_ep = len(keep)
        t = np.arange(obs.shape[0]) // n_ep
        all_ok &= check("phase ticks", n_ep > 0,
                        f"t in [0, {HORIZON - 1}], {n_ep} clean episodes")

        # 物体->目标 = place_target_rel - right_obj_rel
        # （两者都是"X - 手"的形式，相减正好把"手"消掉）
        obj_to_target = obs[:, PLACE_TARGET_REL] - obs[:, RIGHT_OBJ_REL]

        # 打印各相位边界的平均距离表（全部由观测推出，单位：米）
        print(f"  per-phase distances (from obs, metres):")
        print(f"  {'t':>4} {'phase':>8} {'wrist->obj':>11} {'obj->target':>12}")
        for tc, name in PHASE_CHECKPOINTS.items():
            rows = obs[t == tc]
            if len(rows) == 0:
                all_ok &= check(f"t={tc}", False, "no rows at this timestep")
                continue
            d = np.linalg.norm(rows[:, RIGHT_OBJ_REL], axis=-1)         # 腕->物体距离
            dt = np.linalg.norm(obj_to_target[t == tc][:, :2], axis=-1)  # 物体->目标水平距离
            print(f"  {tc:>4} {name:>8} {d.mean():11.3f} {dt.mean():12.3f}")

        # -- 抓取段检查。捏合中心贴上物体时腕->物体距离稳定在 ~0.14 m；
        # 真抓住了的标志是搬运全程这个距离保持恒定。
        d_insert = np.linalg.norm(obs[t == 244][:, RIGHT_OBJ_REL], axis=-1)  # INSERT 时刻
        d_lower = np.linalg.norm(obs[t == 584][:, RIGHT_OBJ_REL], axis=-1)   # LOWER 时刻
        # 检查：INSERT 时腕已足够接近物体（p95 分位数 < 工件对应上限）
        all_ok &= check("grasp proximity", float(np.percentile(d_insert, 95)) < prof["insert_prox"],
                        f"wrist->obj at INSERT p95={np.percentile(d_insert, 95):.3f} m "
                        f"(want < {prof['insert_prox']})")
        # 检查：INSERT 到 LOWER 之间距离漂移很小 => 物体是"跟着手走"的
        drift = float(np.percentile(np.abs(d_lower - d_insert), 95))
        all_ok &= check("rigid hold through CARRY", drift < prof["hold_drift"],
                        f"|d_LOWER - d_INSERT| p95={drift:.3f} m (want < {prof['hold_drift']}, "
                        f"i.e. object moves WITH the hand)")

        # -- 放置段检查。物体最终落在目标上；手撤离远去。
        dt_end = np.linalg.norm(obj_to_target[t == HORIZON - 1][:, :2], axis=-1)
        all_ok &= check("placed on target", float(np.percentile(dt_end, 95)) < 0.05,
                        f"obj->target xy at END p95={np.percentile(dt_end, 95):.3f} m (want < 0.05)")
        # 还握着物体的手停在 ~0.14 m 处，阈值取 0.17：既对 0.14 保有可靠余量，又能
        # 容忍 DART 加噪采集时撤离姿态的抖动（加噪运行最低到 0.176 左右）。
        d_end = np.linalg.norm(obs[t == HORIZON - 1][:, RIGHT_OBJ_REL], axis=-1)
        all_ok &= check("hand retreated", float(np.percentile(d_end, 5)) > 0.17,
                        f"wrist->obj at END p5={np.percentile(d_end, 5):.3f} m (want > 0.17: released, not stuck at ~0.14)")

        # -- 顺序无关的手臂/手指检查（每块内部按关节树顺序、左右交错，
        # 所以只数"有几个维度在动"，不关心具体是哪几维）。
        arm_moving = (np.abs(act[:, ARM_ACT]).max(axis=0) > 0.05).sum()   # 幅度超过 0.05 视为"在动"
        all_ok &= check("one arm parked", arm_moving == 7,
                        f"{arm_moving}/14 arm dims move (want exactly 7: right arm only)")
        pre_moving = (np.abs(act[t < 244][:, FINGER_ACT]).max(axis=0) > 0.05).sum()   # CLOSE 前就在动的手指维数
        finger_closing = (np.abs(act[t == 404][:, FINGER_ACT]).max(axis=0) > 0.1).sum()  # LIFT 时闭合的手指维数
        f_reopen = np.abs(act[t == HORIZON - 1][:, FINGER_ACT]).max()     # 结束帧手指动作的最大幅度
        n_pre = prof["preclose_dims"]
        all_ok &= check("pre-grasp fingers", pre_moving == n_pre,
                        f"{pre_moving}/14 finger dims move before CLOSE (want exactly {n_pre}: "
                        f"{'nut threads the hole index+middle half-closed' if n_pre else 'fully open approach'})")
        all_ok &= check("one hand closed in LIFT", finger_closing == 6,
                        f"{finger_closing}/14 finger dims closed at LIFT (want exactly 6: right hand only, "
                        f"thumb_0 abduction stays 0 by design)")
        all_ok &= check("fingers reopened at end", f_reopen < 0.05,
                        f"max |finger action| at END={f_reopen:.3f} (want < 0.05)")

        # 信息行（不判 OK/BAD）：手臂动作最大幅度。动作单位是 2*弧度
        # （动作 = 2*q_desired），仅供人工核对数值量级是否合理。
        ra = np.abs(act[:, ARM_ACT]).max()
        print(f"  info: max |arm action|={ra:.2f} (action units = 2*rad)")

    print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED - see [BAD] lines above'}")


if __name__ == "__main__":
    main()
