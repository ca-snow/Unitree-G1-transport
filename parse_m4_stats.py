"""
挖掘 M4 里程碑 --repeat 批量实验的日志：按"结局"分组统计每次迭代的
停靠误差 / 抓取质量，定位失败的任务到底是在哪个环节把工件弄丢的。


输出:对每个日志文件，按结局（PLACED 成功放置 / PUSHED 推入 /
NOT PLACED 未放好 / DROPPED 途中掉落）分组，给出各项误差的
均值±标准差 [最小,最大]，并扫描"抓握宽度阈值"看它能否预测失败。

"""
import re
import statistics
import sys


def fresh():
    """返回一次迭代的空白记录（所有统计量初始化为 None / 0）。

    字段含义：
      fwd / lat  —— A 桌抓取点的前向 / 侧向对准误差（厘米）
      yaw        —— 停靠时机器人朝向误差（度）
      retries    —— 停靠重试（后退再靠）次数
      grip       —— M3 抓取冻结时刻的抓握宽度（米，越大说明捏得越松）
      lift       —— 同一时刻工件被抬起的高度（厘米）
      bfwd / blat—— B 桌放置区的前向 / 侧向误差（厘米）
      drop_t     —— 工件掉落时刻（秒），没掉则为 None
      cut_t      —— M3 抓取冻结（CUT）时刻（秒）
    """
    return {"fwd": None, "lat": None, "yaw": None, "retries": 0,
            "grip": None, "lift": None, "bfwd": None, "blat": None,
            "drop_t": None, "cut_t": None}


def parse(path):
    #逐行解析一个日志文件，返回"每次迭代一条记录"的列表。

    #做法：维护一个"当前迭代"字典 cur，逐行用正则匹配感兴趣的日志行、
    #往 cur 里填数；遇到迭代结束行（"--repeat: iteration ... done"）时
    #把 cur 收进结果列表并重新开一条空记录。
    
    iters = []      # 已完成迭代的记录列表
    cur = fresh()   # 当前正在累积的迭代记录
    # 每种日志行对应一个正则；正则内容必须与 transport_demo.py 的打印格式逐字符一致
    pat = {
        "fwd": re.compile(r"piece fwd err:\s*([+-]?[\d.]+) cm"),    # 抓取点前向误差
        "lat": re.compile(r"piece lat err:\s*([+-]?[\d.]+) cm"),    # 抓取点侧向误差
        "yaw": re.compile(r"dock yaw err :\s*([+-]?[\d.]+) deg"),   # 停靠朝向误差
        "bfwd": re.compile(r"zone fwd err :\s*([+-]?[\d.]+) cm"),   # 放置区前向误差
        "blat": re.compile(r"zone lat err :\s*([+-]?[\d.]+) cm"),   # 放置区侧向误差
        # M3 CUT 行：抓取冻结时刻，同时带出抓握宽度（米）和抬升高度（厘米）
        "cut": re.compile(r"t=\s*([\d.]+)s M3 CUT at bc t=\d+: grasp frozen "
                          r"\(grip ([\d.]+) m, lift ([+-][\d.]+) cm"),
        # DROPPED 行：工件在搬运途中掉落的时刻
        "drop": re.compile(r"t=\s*([\d.]+)s JOB \d+ DROPPED"),
        # 迭代结束行：捕获迭代号、耗时和结局描述文字
        "end": re.compile(r"--repeat: iteration (\d+)/\d+ done in ([\d.]+) s: (.+?) ====="),
    }
    # errors="replace"：日志里偶发的坏字节替换成占位符，不让解析中断
    for line in open(path, encoding="utf-8", errors="replace"):
        m = pat["fwd"].search(line)
        if m:
            cur["fwd"] = float(m.group(1))
            continue
        m = pat["lat"].search(line)
        if m:
            cur["lat"] = float(m.group(1))
            continue
        m = pat["yaw"].search(line)
        if m:
            cur["yaw"] = float(m.group(1))
            continue
        m = pat["bfwd"].search(line)
        if m:
            cur["bfwd"] = float(m.group(1))
            continue
        m = pat["blat"].search(line)
        if m:
            cur["blat"] = float(m.group(1))
            continue
        # 停靠重试行不需要正则，直接子串判断即可
        if "RETRY #" in line and "backing off" in line:
            cur["retries"] += 1
            continue
        m = pat["cut"].search(line)
        if m:
            cur["cut_t"] = float(m.group(1))
            cur["grip"] = float(m.group(2))
            cur["lift"] = float(m.group(3))
            continue
        m = pat["drop"].search(line)
        if m:
            cur["drop_t"] = float(m.group(1))
            continue
        m = pat["end"].search(line)
        if m:
            # 迭代结束：把结局描述文字归一化成四种"判决"（verdict）
            v = m.group(3)
            if "DROPPED" in v:
                verdict = "DROPPED"        # 搬运途中掉落
            elif v.startswith("NOT PLACED"):
                verdict = "NOT PLACED"     # 到了 B 桌但没放进目标区
            elif v.startswith("PLACED-BY-PUSH"):
                verdict = "PUSHED"         # 靠推挤才进目标区（不算干净的成功）
            else:
                verdict = "PLACED"         # 正常放置成功
            # 结局文字里若带 "(x.x cm)"，是物体最终位置到目标点的水平距离
            mm = re.search(r"\(([\d.]+) cm\)", v)
            cur["dxy"] = float(mm.group(1)) if mm else None
            cur["iter"] = int(m.group(1))
            cur["verdict"] = verdict
            iters.append(cur)   # 本次迭代收尾入列
            cur = fresh()       # 开始累积下一次迭代
    return iters


def stat(vals):
    #把一组数值汇总成 "均值±标准差 [最小,最大] n=样本数" 的字符串。

    #None（该迭代没打出这一项）会被剔除；全为 None 时返回 "-"。
    #标准差用总体标准差pstdev（样本只有1个时记 0，避免除零）。
    
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    mu = statistics.mean(vals)
    sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return f"{mu:+.2f}±{sd:.2f} [{min(vals):+.1f},{max(vals):+.1f}] n={len(vals)}"


def report(path):
    #解析一个日志文件并打印按结局分组的统计报告。
    iters = parse(path)
    print(f"\n================ {path}: {len(iters)} iterations ================")
    # 按结局分组
    groups = {}
    for it in iters:
        groups.setdefault(it["verdict"], []).append(it)
    # 按组内数量从多到少排序输出
    for v, g in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"-- {v}: {len(g)}/{len(iters)} ({100 * len(g) / len(iters):.0f}%)")
        print(f"   pick fwd err (cm): {stat([i['fwd'] for i in g])}")
        print(f"   pick lat err (cm): {stat([i['lat'] for i in g])}")
        print(f"   pick yaw err (deg): {stat([i['yaw'] for i in g])}")
        print(f"   grip @cut (m)    : {stat([i['grip'] for i in g])}")
        print(f"   lift @cut (cm)   : {stat([i['lift'] for i in g])}")
        print(f"   place fwd err(cm): {stat([i['bfwd'] for i in g])}")
        print(f"   place lat err(cm): {stat([i['blat'] for i in g])}")
        if v == "PLACED":
            # 只有成功组才有"物体到目标点距离"这一项
            print(f"   obj->target (cm) : {stat([i['dxy'] for i in g])}")
        if v == "DROPPED":
            # 掉落组额外统计：从抓取冻结到掉落之间坚持了几秒
            #（掉得早说明根本没抓稳，掉得晚说明是行走颠簸把它甩掉的）
            hold = [i['drop_t'] - i['cut_t'] for i in g
                    if i['drop_t'] is not None and i['cut_t'] is not None]
            print(f"   carry time before drop (s): {stat(hold)}")
        retried = sum(1 for i in g if i["retries"] > 0)
        print(f"   dock retried in {retried}/{len(g)} runs")
    
    placed = [i for i in iters if i["verdict"] == "PLACED"]
    bad = [i for i in iters if i["verdict"] != "PLACED"]
    if placed and bad:
        # 候选阈值 0.150~0.180 米，步长 5 毫米：覆盖实测中"捏紧"到"明显松脱"的抓握宽度范围
        for thr in (0.150, 0.155, 0.160, 0.165, 0.170, 0.175, 0.180):
            tp = sum(1 for i in bad if (i["grip"] or 0) > thr)     # 命中的失败数（真阳性）
            fp = sum(1 for i in placed if (i["grip"] or 0) > thr)  # 误伤的成功数（假阳性）
            print(f"   grip>{thr:.3f}: catches {tp}/{len(bad)} failures, "
                  f"false-alarms {fp}/{len(placed)} placed")

# 主流程：对命令行传入的每个日志文件分别出报告
for p in sys.argv[1:]:
    report(p)
