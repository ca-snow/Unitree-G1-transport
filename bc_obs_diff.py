# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
离线的 110 维 BC 观测差分 + 逐块交换探针

1. 关节顺序校验(JOINT-ORDER CHECK):两套装置的原始 obs 索引是否解析到
   同名的关节(transport 的观测构造器硬编码了 pickplace 的 gather索引;
   此前只断言过关节数量一致，没查过顺序.)
2. 字段差分(FIELD DIFF):逐观测字段(joint_pos / joint_vel / hand /
   left_obj_rel / right_obj_rel / last_action / phase / place_target_rel)
   计算 transport 交接帧与训练参考帧(t=0 和 t=20 两个)差多远。
3. 交换探针(SWAP PROBE):把某一个字段在 transport 帧和训练帧之间互换
   （双向都换），构造"杂交观测"喂给检查点，观察右肘输出。哪个字段一换
   就翻转肘部符号，哪个字段【就是】病因。不靠猜、不靠肉眼。

"""

import argparse

import torch

from bc_train import SharedPolicy

# 110 维观测布局（字段名, 起始维, 结束维），与 pickplace 的 ObservationsCfg
# 以及两个 dump 生成方保持一致。
FIELDS = [
    ("joint_pos",        0,  29),   # 29 个身体关节角
    ("joint_vel",       29,  58),   # 29 个身体关节角速度
    ("hand_joints",     58,  72),   # 14 个手指关节角（Dex3 双手）
    ("left_obj_rel",    72,  75),   # 物体相对左腕的位置向量 (x,y,z)
    ("right_obj_rel",   75,  78),   # 物体相对右腕的位置向量 (x,y,z)
    ("last_action",     78, 106),   # 上一步执行的 28 维动作
    ("phase",          106, 107),   # episode 相位（t/800）
    ("place_target_rel", 107, 110), # 放置目标相对右腕的位置向量
]
# 28 维动作的排列顺序（镜像 BC 环境的 JointPositionActionCfg）：
# 左臂 0-6，右臂 7-13（肩 pitch/roll/yaw、肘=第 10 维、腕 roll/pitch/yaw），
# 双手手指 14-27。
ELBOW_SLOT = 10                   # 右肘在动作向量中的槽位（诊断的核心观察量）
RSP_SLOT = 7                      # 右肩 pitch（right_shoulder_pitch）槽位
ACTION_SCALE = 0.5                # 环境动作缩放：写入的目标关节角 = 0 + 0.5 * 动作


def dim_names(dump):
    #给 110 个观测维度逐一起可读的名字（如 "jp:right_elbow_joint"）。

    #输入：dump 字典（--dump_obs 文件加载结果，含 body_names/hand_names）。
    #输出：长度 110 的字符串列表，顺序与观测维度一一对应。
   
    body, hand = dump["body_names"], dump["hand_names"]
    xyz = ["x", "y", "z"]
    names = []
    names += [f"jp:{n}" for n in body]                     # 0:29 关节角
    names += [f"jv:{n}" for n in body]                     # 29:58 关节角速度
    names += [f"hand:{n}" for n in hand]                   # 58:72 手指
    names += [f"left_obj_rel:{a}" for a in xyz]            # 72:75
    names += [f"right_obj_rel:{a}" for a in xyz]           # 75:78
    names += [f"last_action[{i}]" for i in range(28)]      # 78:106
    names += ["phase"]                                     # 106
    names += [f"place_target_rel:{a}" for a in xyz]        # 107:110
    return names


def row_at(dump, tick):
    #在 dump 里找步号 tick 对应的记录行。

    #返回 (行索引, 实际命中的 tick)。tick<0 表示取第一条记录（即交接帧）；
    #若指定的 tick 没被记录（transport 每 2 个 tick 才记一行），则取最近的已记录 tick。
    
    ticks = list(dump["tick"])
    if tick < 0:                                  # -1 = 第一条记录行
        return 0, ticks[0]
    if tick in ticks:
        return ticks.index(tick), tick
    # 找最近的已记录 tick（transport 每隔 2 个 tick 记录一次）
    j = min(range(len(ticks)), key=lambda i: abs(ticks[i] - tick))
    return j, ticks[j]


def elb(policy, obs):
    #把单帧观测 obs [110] 喂给策略，返回 (右肘动作, 右肩pitch动作) 两个标量。
    with torch.no_grad():
        a = policy(obs.unsqueeze(0))[0]
    return a[ELBOW_SLOT].item(), a[RSP_SLOT].item()


def main():
    #诊断入口：加载两份 dump + 检查点，依次执行关节序校验、闭环轨迹对照、字段差分、交换探针。
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # --play_dump：bc_play.py --dump_obs 生成的训练环境参考 dump
    ap.add_argument("--play_dump", required=True)
    # --demo_dump：transport_demo.py --dump_obs 生成的部署侧 dump
    ap.add_argument("--demo_dump", required=True)
    # --checkpoint：bc_train.py 输出的 .pt 检查点（用来做交换探针的前向）
    ap.add_argument("--checkpoint", required=True)
    # --play_ticks：作为"训练参考帧"的步号列表；默认 [0, 20] —— t=0 是
    # 起始帧，t=20 已进入运动早期，两个参考点可区分"静态差异"和"演化差异"
    ap.add_argument("--play_ticks", type=int, nargs="+", default=[0, 20],
                    help="Trained reference ticks to compare/swap against.")
    # --demo_tick：要分析的 transport 步号（-1 = 第一条记录，即交接帧）
    ap.add_argument("--demo_tick", type=int, default=-1,
                    help="Transport tick to analyse (-1 = the handover frame).")
    # --top：每个字段打印差异最大的前几个维度
    ap.add_argument("--top", type=int, default=4, help="Top differing dims per field.")
    args = ap.parse_args()

    # 加载两份 dump 与检查点（纯 CPU 即可，网络很小）
    play = torch.load(args.play_dump, map_location="cpu", weights_only=False)
    demo = torch.load(args.demo_dump, map_location="cpu", weights_only=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    policy = SharedPolicy(ckpt["obs_dim"], ckpt["act_dim"])
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"[diff] play: {args.play_dump} ({play['obs'].shape[0]} rows, {play.get('workpiece')})")
    print(f"[diff] demo: {args.demo_dump} ({demo['obs'].shape[0]} rows, {demo.get('workpiece')})"
          f" handover={demo.get('handover')}")

    # ---- 1. 关节顺序校验 ----------------------------------------------
    # 两份 dump 都记录了"每个 obs 索引解析出的关节名"，逐位置对比：
    # 任何一个位置的名字不同，都意味着 transport 的观测被打乱了顺序。
    ok = True
    for key in ("body_names", "hand_names"):
        a, b = list(play[key]), list(demo[key])
        if a == b:
            print(f"[diff] JOINT-ORDER CHECK {key}: PASS ({len(a)} joints identical)")
        else:
            ok = False
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"[diff] JOINT-ORDER MISMATCH {key}[{i}]: play={x} demo={y}")
    if not ok:
        # 顺序错位时后面所有分析都不可信，先修索引表再看
        print("[diff] !!! obs indices resolve to DIFFERENT joints - the transport obs are "
              "scrambled; fix the index tables before reading anything else below")

    # ---- 0. 两边记录的闭环轨迹逐 tick 对照：分歧发生在哪一拍 ----
    #按 tick展示分歧点。
    print("\n[diff] recorded closed-loop rows (elb = raw action, rel_z = right_obj_rel z):")
    print("[diff]   tick |  play elb  demo elb |  play rel_z  demo rel_z |  play phase  demo phase")
    pticks = list(play["tick"])
    dticks = [int(t) for t in demo["tick"]]
    for tk in dticks:
        if tk > 60 or tk not in pticks:   # 只看前 60 拍，且两边都有记录的 tick
            continue
        pi, di = pticks.index(tk), dticks.index(tk)
        po, do_ = play["obs"][pi], demo["obs"][di]
        pa, da = play["act"][pi], demo["act"][di]
        # 两边肘部动作差超过 0.05 即标记为分歧点（SPLIT）
        mark = "  <-- SPLIT" if abs(pa[ELBOW_SLOT] - da[ELBOW_SLOT]) > 0.05 else ""
        print(f"[diff]   {tk:4d} |  {pa[ELBOW_SLOT]:+.3f}    {da[ELBOW_SLOT]:+.3f}  |"
              f"  {po[77]:+.3f}      {do_[77]:+.3f}   |"
              f"  {po[106]:.4f}      {do_[106]:.4f}{mark}")

    names = dim_names(play)
    di, dt = row_at(demo, args.demo_tick)
    d_obs, d_act = demo["obs"][di], demo["act"][di]
    print(f"\n[diff] demo frame: tick {dt} | recorded action: "
          f"elb={d_act[ELBOW_SLOT]:+.3f} (tgt {ACTION_SCALE * d_act[ELBOW_SLOT]:+.3f})")

    # ---- 2. 逐字段差分 -------------------------------------------------
    # 对每个训练参考帧：计算 |demo 观测 - play 观测|，按字段汇报最大值
    # （Linf）、平均值，以及差异最大的前 --top 个维度（带关节名）。
    for ptick in args.play_ticks:
        pi, pt = row_at(play, ptick)
        p_obs, p_act = play["obs"][pi], play["act"][pi]
        diff = (d_obs - p_obs).abs()
        print(f"\n[diff] ===== demo tick {dt}  vs  play tick {pt} "
              f"(play action: elb={p_act[ELBOW_SLOT]:+.3f}) =====")
        for fname, lo, hi in FIELDS:
            d = diff[lo:hi]
            order = torch.argsort(d, descending=True)[: args.top]  # 该字段内差异最大的维度
            top = ", ".join(f"{names[lo + i]}={d[i]:.3f}"
                            f"({d_obs[lo + i]:+.3f} vs {p_obs[lo + i]:+.3f})"
                            for i in order.tolist() if d[i] > 1e-4)
            print(f"[diff]   {fname:17s} Linf={d.max():.4f} mean={d.mean():.4f}"
                  f"{'  top: ' + top if top else ''}")

    # ---- 3. 交换探针 ----------------------------------------------------
    # 对每个字段做双向交换：
    # "治愈"（heal）：demo 观测里只把这一个字段换成 play 的值——如果肘部输出因此回到训练侧的符号，说明该字段是病因；
    # "投毒"（poison）：play 观测里只把这一个字段换成 demo 的值——如果肘部因此复现下扎，进一步确认。
    for ptick in args.play_ticks:
        pi, pt = row_at(play, ptick)
        p_obs = play["obs"][pi]
        e0, s0 = elb(policy, d_obs)   # 基线 1：原封不动的 demo 观测（下扎侧）
        e1, s1 = elb(policy, p_obs)   # 基线 2：原封不动的 play 观测（训练侧）
        print(f"\n[diff] ===== SWAP PROBE vs play tick {pt} "
              f"(elbow target = 0.5 x raw; NEGATIVE = raise, POSITIVE = dive) =====")
        print(f"[diff]   demo obs unchanged      -> elb {e0:+.3f} (tgt {ACTION_SCALE * e0:+.3f}) "
              f"rshpitch {s0:+.3f}   <- the DIVE baseline")
        print(f"[diff]   play obs unchanged      -> elb {e1:+.3f} (tgt {ACTION_SCALE * e1:+.3f}) "
              f"rshpitch {s1:+.3f}   <- the trained reference")
        for fname, lo, hi in FIELDS:
            h1 = d_obs.clone(); h1[lo:hi] = p_obs[lo:hi]      # "治愈"：demo 帧植入 play 的该字段
            h2 = p_obs.clone(); h2[lo:hi] = d_obs[lo:hi]      # "投毒"：play 帧植入 demo 的该字段
            eh, _ = elb(policy, h1)
            ep, _ = elb(policy, h2)
            # 判定"治愈"：基线 demo 是下扎(e0>0)，换字段后肘部回到 0 以下
            # 且不高于训练值的一半（即真正回到训练侧，不只是轻微好转）
            flag_h = "  <== HEALS the dive" if (e0 > 0 and eh < min(0.0, 0.5 * e1)) else ""
            # 判定"投毒"：基线 play 是抬起(e1<0)，换字段后肘部翻成正（下扎）
            flag_p = "  <== POISONS the raise" if (e1 < 0 and ep > 0) else ""
            print(f"[diff]   demo, {fname:17s}<-play -> elb {eh:+.3f}{flag_h}")
            print(f"[diff]   play, {fname:17s}<-demo -> elb {ep:+.3f}{flag_p}")


    # 训练流形/off-manifold）——那就是"必须重新采集数据"的结论。
    print("\n[diff] reading guide: a field is THE cause iff healing it on the demo side "
          "moves the elbow to the trained sign AND poisoning it on the play side "
          "reproduces the dive. If no single field does both, the cause is spread "
          "across fields (aggregate off-manifold state) - that is the re-collect verdict.")


if __name__ == "__main__":
    main()
