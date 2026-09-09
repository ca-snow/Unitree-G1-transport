"""
第 3 轮 M4 批量日志分析:逐迭代提取"最后一次被接受的抓取停靠
(M1 报告)"与最终判定(PLACED/DROPPED/NOT PLACED)的对应关系。
"""
import re
import statistics
from collections import Counter

def parse(path):
    """把一个日志文件解析成"迭代记录"列表。

    每条记录字段:
      m1        - 该迭代内出现过的所有 M1 报告(kind/fwd/lat/yaw)
      last_pass - 最后一次 PASS 的 M1 报告(即被接受的停靠数据)
      last_fail - 最后一次 FAIL 的 M1 报告
      retries   - M1 门限重试的最大次数
      exhausted - 是否耗尽重试预算、带着不合格停靠继续
      cut       - M3 CUT 时刻的抓握距离/抬升高度
      drop      - 迭代内是否发生过掉件
      verd      - 迭代最终判定(PLACED/DROPPED/NOT PLACED)
    """
    iters = []
    cur = {
        "m1": [], "cut": None, "drop": None, "end": None,
        "retries": 0, "last_pass": None, "last_fail": None,
        "exhausted": False,
    }
    # M1 报告的三行数值:前后误差、左右误差、停靠偏航误差(带门限范围)
    m1_re = re.compile(
        r"piece fwd err:\s*([+-]?[\d.]+) cm \(gate ([+-]?[\d.]+)\.\.([+-]?[\d.]+).*\n"
        r"\[demo\]   piece lat err:\s*([+-]?[\d.]+) cm \(gate ([+-]?[\d.]+)\.\.([+-]?[\d.]+).*\n"
        r"\[demo\]   dock yaw err :\s*([+-]?[\d.]+) deg \(gate ([+-]?[\d.]+)\.\.([+-]?[\d.]+)",
        re.M,
    )
    header = re.compile(r"===== M1 REPORT \((PASS|FAIL)\) =====")
    # M3 CUT 行:抓握冻结时的手-物距离和抬升高度
    cut = re.compile(r"M3 CUT at bc t=\d+: grasp frozen \(grip ([\d.]+) m, lift ([+-][\d.]+) cm")
    drop = re.compile(r"JOB \d+ DROPPED")
    # 迭代结束行(--repeat 模式):迭代号、用时、判定文本
    end = re.compile(r"--repeat: iteration (\d+)/\d+ done in ([\d.]+) s: (.+?) =====")
    retry = re.compile(r"M1 gate failed - RETRY #(\d+)")
    text = open(path, encoding="utf-8", errors="replace").read()
    # 逐行扫描,保持事件的先后顺序
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = header.search(line)
        if m:
            kind = m.group(1)
            # M1 报告的数值行紧跟在标题后面,取 6 行窗口做正则匹配
            block = "\n".join(lines[i:i+6])
            mm = m1_re.search(block)
            if mm:
                rec = {
                    "kind": kind,
                    "fwd": float(mm.group(1)),
                    "lat": float(mm.group(4)),
                    "yaw": float(mm.group(7)),
                }
                cur["m1"].append(rec)
                if kind == "PASS":
                    cur["last_pass"] = rec
                else:
                    cur["last_fail"] = rec
            i += 1
            continue
        m = retry.search(line)
        if m:
            cur["retries"] = max(cur["retries"], int(m.group(1)))
            i += 1
            continue
        # 耗尽重试预算、带着当前停靠硬着头皮继续的标志行
        if "proceeding with current dock" in line or "retry budget" in line.lower():
            cur["exhausted"] = True
            i += 1
            continue
        m = cut.search(line)
        if m:
            cur["cut"] = {"grip": float(m.group(1)), "lift": float(m.group(2))}
            i += 1
            continue
        if drop.search(line):
            cur["drop"] = True
            i += 1
            continue
        m = end.search(line)
        if m:
            # 迭代收尾:从判定文本归一化出三种结果之一
            v = m.group(3)
            if "DROPPED" in v:
                verd = "DROPPED"
            elif v.startswith("NOT PLACED"):
                verd = "NOT PLACED"
            else:
                verd = "PLACED"
            cur["verd"] = verd
            cur["iter"] = int(m.group(1))
            iters.append(cur)
            cur = {
                "m1": [], "cut": None, "drop": None, "end": None,
                "retries": 0, "last_pass": None, "last_fail": None,
                "exhausted": False,
            }
        i += 1
    return iters


def stat(vals):
    #排版一组数值的统计:均值±标准差 [最小,最大] n=样本数。
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{statistics.mean(vals):+.2f}±{sd:.2f} [{min(vals):+.1f},{max(vals):+.1f}] n={len(vals)}"


def report(path):
    #打印一个日志文件的汇总:判定分布、重试直方图,再按判定分组统计"最后一次 PASS 的停靠数据"的分布。
    iters = parse(path)
    print(f"\n======== {path}: {len(iters)} iters ========")
    print("verdicts", Counter(it["verd"] for it in iters))
    print("M1-gate retries hist", Counter(it["retries"] for it in iters))
    print("exhausted", sum(1 for it in iters if it["exhausted"]))
    for verd in ("PLACED", "DROPPED", "NOT PLACED"):
        g = [it for it in iters if it["verd"] == verd]
        if not g:
            continue
        print(f"-- {verd} {len(g)}")
        print(f"   last PASS fwd {stat([it['last_pass']['fwd'] if it['last_pass'] else None for it in g])}")
        print(f"   last PASS lat {stat([it['last_pass']['lat'] if it['last_pass'] else None for it in g])}")
        print(f"   last PASS yaw {stat([it['last_pass']['yaw'] if it['last_pass'] else None for it in g])}")
        print(f"   M1 retries    {stat([float(it['retries']) for it in g])}")
        n_pass = sum(1 for it in g if it["last_pass"])
        n_only_fail = sum(1 for it in g if it["last_pass"] is None)
        print(f"   accepted via PASS {n_pass}/{len(g)}; proceeded without PASS {n_only_fail}")
        # 统计掉件迭代里,最后一次 PASS 落在"危险半区"(偏深 / 偏左
        # 超过 0.8 cm / 偏航为正)的比例
        if verd == "DROPPED":
            danger = 0
            for it in g:
                p = it["last_pass"]
                if not p:
                    danger += 1
                    continue
                if p["fwd"] > 0 or p["lat"] > 0.8 or p["yaw"] > 0:
                    danger += 1
            print(f"   last-pass in deep/left/yaw+ region: {danger}/{len(g)}")


for p in ("m4_round3_job5.log", "m4_round3_job6.log"):
    report(p)
