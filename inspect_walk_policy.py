# Copyright (c) 2025, Unitree RL Training Layer.
# SPDX-License-Identifier: Apache-2.0

"""
打印行走（locomotion）策略 checkpoint 文件的内部结构——"行走+抓取"集成工作的第 0 步。
"""

import argparse

import torch

def _describe_state_dict(sd: dict, indent: str = "  ") -> None:
    #逐行打印一个 state_dict（参数名 -> 张量形状），并推断网络的输入/输出维度。

    #原理：全连接层（Linear）的权重是二维张量 [输出维, 输入维]。按保存顺序，
    #第一个二维权重的"输入维"就是观测维度，最后一个二维权重的"输出维"就是动作维度。
    
    lin_shapes = []  # 收集所有二维权重（即 Linear 层的 weight）
    for name, t in sd.items():
        shape = tuple(t.shape) if hasattr(t, "shape") else "?"
        print(f"{indent}{name}: {shape}")
        if hasattr(t, "shape") and len(t.shape) == 2:
            lin_shapes.append((name, tuple(t.shape)))
    if lin_shapes:
        first_name, first = lin_shapes[0]
        last_name, last = lin_shapes[-1]
        # first[1]：第一层权重的第二维 = 网络输入维度
        print(f"{indent}--> first Linear  {first_name}: in={first[1]}")
        # last[0]：最后一层权重的第一维 = 网络输出维度
        print(f"{indent}--> last  Linear  {last_name}: out={last[0]}")


def main() -> None:
    #加载 checkpoint 并按格式分情况打印其结构。
    parser = argparse.ArgumentParser(description="Inspect a locomotion policy checkpoint.")
    # --checkpoint：要检查的 .pt 权重文件路径（必填）
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    # --- 第一步：先尝试按 TorchScript 格式加载（rsl_rl 的 `export` 或 jit.save 的产物）---
    try:
        module = torch.jit.load(args.checkpoint, map_location="cpu")
        print(f"[format] TorchScript module: {type(module).__name__}")
        print("[graph inputs]")
        # 打印计算图的输入签名（可看出模型期望的输入张量类型/形状）
        for inp in module.graph.inputs():
            print(f"  {inp.debugName()}: {inp.type()}")
        print("[parameters]")
        _describe_state_dict(dict(module.named_parameters()))
        return  # TorchScript 分支处理完毕，直接结束
    except Exception:
        pass  # 不是 TorchScript，继续走普通 checkpoint 分支

    # --- 第二步：按原始 torch checkpoint 加载 ---
    # weights_only=False：checkpoint 里可能带非张量对象（训练信息等），需要完整反序列化。
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"[format] torch.load -> {type(ckpt).__name__}")

    if isinstance(ckpt, dict):
        print(f"[top-level keys] {list(ckpt.keys())}")
        for key in ("model_state_dict", "policy", "state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                print(f"[{key}]")
                _describe_state_dict(ckpt[key])
                break
        else:
            if all(hasattr(v, "shape") for v in ckpt.values()):
                print("[state_dict]")
                _describe_state_dict(ckpt)
            else:
                # 兜底：找出所有"值全是张量的嵌套 dict"，逐个当 state_dict 打印
                for k, v in ckpt.items():
                    if isinstance(v, dict) and v and all(hasattr(t, "shape") for t in v.values()):
                        print(f"[{k}] (nested state_dict)")
                        _describe_state_dict(v)
        # 归一化器统计量非常关键：如果 checkpoint 里存了 norm/rms（均值方差等），
        # 说明训练时观测被归一化过——推理时适配器必须用这些"同一套"统计量
        for k in ckpt.keys() if isinstance(ckpt, dict) else []:
            if "norm" in str(k).lower() or "rms" in str(k).lower():
                print(f"[NOTE] normalizer entry found: {k} - the adapter must apply it")
    else:
        print(ckpt)


if __name__ == "__main__":
    main()
