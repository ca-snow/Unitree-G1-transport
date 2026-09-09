"""
注意：本脚本只做纯粹的 np.concatenate 行拼接，不打乱、不去重、不做任何
校验之外的处理；数据体检请用 dataset_check.py。

"""
from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    #按命令行顺序加载各输入 .npz，沿第 0 维（行）拼接后写出合并文件。
    parser = argparse.ArgumentParser(description=__doc__)
    # inputs：一个或多个输入 .npz 文件（位置参数，按给出的顺序拼接）
    parser.add_argument("inputs", nargs="+", help="input .npz files (obs/actions)")
    # --out：合并后的输出 .npz 路径
    parser.add_argument("--out", required=True, help="merged output .npz")
    args = parser.parse_args()

    obs_parts, act_parts = [], []
    for path in args.inputs:
        data = np.load(path)
        obs, act = data["obs"], data["actions"]
        # obs 与 actions 必须一一配对（行数一致），否则文件已损坏
        assert obs.shape[0] == act.shape[0], f"{path}: obs/actions row mismatch"
        print(f"  {path}: {obs.shape[0]} rows (obs {obs.shape[1]}, act {act.shape[1]})")
        obs_parts.append(obs)
        act_parts.append(act)
    # 沿行方向拼接：多个文件首尾相接，episode 内部顺序保持不变
    obs_all = np.concatenate(obs_parts, axis=0)
    act_all = np.concatenate(act_parts, axis=0)
    np.savez(args.out, obs=obs_all, actions=act_all)
    print(f"-> {args.out}: {obs_all.shape[0]} rows total")


if __name__ == "__main__":
    main()
