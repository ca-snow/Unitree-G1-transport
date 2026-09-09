# Unitree G1 人形机器人「行走 + 抓取 + 搬运」全流程仿真项目


基于 **NVIDIA Isaac Lab** 与 **Unitree 官方仿真层(unitree_sim_isaaclab)** 的 G1(29 自由度 + Dex3 灵巧手)工件搬运演示:

> 机器人从工作台 A 依次识别并抓取 7 个工件(方块 / 螺栓 / 螺母 / 电钻),行走搬运至工作台 B 指定分区放置,全程连续循环。
> 行走由 **PPO 强化学习** 策略驱动,抓取由 **脚本化专家 + 行为克隆(BC)** 策略完成,顶层由状态机编排
> 附带 **YOLO 视觉分类** 与视觉误差影子测量。

## 1. 项目结构

```
unitree_RL/
├── README.md
├── LICENSE                        # Apache-2.0
├── .gitignore
│
│  ── 场景与资产准备 ──
├── measure_asset.py               # 测量 USD 资产包围盒/单位/物理 API,选缩放系数
├── prepare_table_asset.py         # 从带黄框的桌子资产派生干净版桌子 USD
├── prepare_drill_asset.py         # 给纯视觉 YCB 电钻资产补刚体/凸分解碰撞体
├── prepare_factory_asset.py       # 剥离 Factory 螺栓资产里的装配关节(否则回弹原点)
├── view_transport_scene.py        # 交互式查看双桌搬运场景布局(无训练)
│
│  ── 行走(浮动基座 RL)──
├── inspect_walk_policy.py         # 离线解析行走 checkpoint 的格式与网络输入输出维度
│
│  ── 抓取(固定基座 BC 流水线)──
├── inspect-robot.py               # 一次性打印关节名/限位/连杆名(写专家脚本的依据)
├── grasp-expert.py                # 脚本化抓取专家(离线 IK 规划+关节空间执行)+ 数据采集
├── dataset_check.py               # 离线体检专家数据集(obs/action 是否描述预期循环)
├── bc_dataset_merge.py            # 合并多次采集的 .npz 数据集
├── bc_train.py                    # 行为克隆训练器(网络结构与 skrl PPO 模型一致)
├── bc_play.py                     # BC 策略回放评估(成功率+录像)
├── bc_obs_diff.py                 # 110 维观测逐块对比诊断(排查换环境后策略失效)
├── bc_rollout_diag.py             # 离线闭环诊断(排查"最后动作捷径"过拟合)
├── finetune_from_bc.py            # 以 BC 权重为起点的 PPO 微调(可选)
├── train-catch.py                 # skrl PPO 训练入口(抓取任务变体)
├── play-catch.py                  # skrl checkpoint 回放录像
├── play_rl.py                     # PPO 微调后 checkpoint 的评估入口
│
│  ── 整合搬运演示(核心)──
├── transport_demo.py              # 主演示:行走+抓取+搬运状态机、M1~M4 里程碑评分、
│                                  #   视觉挂钩、双桌录像、批量统计,全部 CLI 开关
├── parse_m4_stats.py              # 挖掘 --repeat 批量日志:停靠误差/抓取质量 vs 结果
├── parse_m4_round3.py             # 第 3 轮批量日志:最后接受的停靠数据与掉件的关联
│
│  ── 视觉层(YOLO 类别闸门 + 影子测量)──
├── head_cam_calib.py              # 头部相机标定:视场覆盖/投影链误差/表面-根偏移
├── vision_gate.py                 # 视觉模块:相机数学、影子日志、YOLO 闸门、实时浮窗
├── make_yolo_dataset.py           # 仿真内自动标注 YOLO 数据集生成(域随机化)
├── make_vision_report.py          # 影子 CSV → 面试/评审用误差对比报告(Markdown)
│
│  ── 任务定义包(部署到 unitree_sim_isaaclab/tasks/ 下)──
└── tasks/rl/
    ├── g1_locomotion/             # 行走任务:平地速度跟踪 RL 环境
    │   ├── locomotion_env_cfg.py  #   环境配置(动作=腿+左臂,右臂预留给抓取)
    │   └── agents/skrl_ppo_cfg.yaml
    ├── g1_pickplace/              # 抓取任务:固定基座 pick-and-place 环境
    │   ├── pickplace_env_cfg.py   #   环境配置(28 维关节位置动作,110 维观测)
    │   ├── mdp/                   #   observations / rewards / terminations / events
    │   └── agents/skrl_ppo_cfg.yaml
    └── g1_transport/              # 搬运场景:双桌+分区标记+工件布局(纯场景)
        └── transport_scene_cfg.py
```

## 2. 依赖与部署

### 2.0 本项目使用阿里云 无影AI云电脑企业版(新加坡) 进行开发，直接使用[具身智能_AI仿真核心包]镜像
    预装环境信息：OS：Ubuntu 22.04.5 LTS  驱动：CUDA 12.8  Pytorch 2.9.1
                  Isaacc Sim 5.1.0、Isaac Lab 2.3.0 
    机器配置根据个人需求选择，镜像详细信息请到阿里云官方相应页面了解
    (境外站点实例需要另外创建办公网络绑定到实例使用)
    
### 2.1 前置安装

| 组件 | 说明 |
|---|---|
| [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab) | Unitree 官方仿真层,提供 G1+Dex3 的 USD 模型与执行器配置(`G1RobotPresets`),Apache-2.0,**需单独克隆** |

> ⚠️ 不要向 Isaac 的 Python 环境安装 GUI 版 `opencv-python`,会破坏 kit 扩展环境;`vision_gate.py` 的浮窗在 headless cv2 下自动退回 omni.ui 渲染。

### 2.2 文件部署位置(两处)

本仓库的文件在运行机上分属两个目录,**必须按下表放置**:

| 仓库内容 | 部署到 | 原因 |
|---|---|---|
| 根目录所有 `*.py` 脚本 | `<IsaacLab>/scripts/reinforcement_learning/skrl/` | 统一用 `./isaaclab.sh -p` 启动;`transport_demo.py`、`make_yolo_dataset.py` 等与 `vision_gate.py` 必须同目录 |
| `tasks/rl/`(整个包) | `<unitree_sim_isaaclab>/tasks/rl/` | 任务包要 `from tasks.common_config import G1RobotPresets`,必须与官方 `tasks` 包同命名空间 |

```bash
# 示例部署脚本
IL=/opt/NVIDIA/IsaacLab
USIM=$HOME/unitree/unitree_sim_isaaclab

git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git "$USIM"
git clone https://github.com/ca-snow/Unitree-G1-transport.git

cp unitree_RL/*.py        "$IL/scripts/reinforcement_learning/skrl/"
cp -r unitree_RL/tasks/rl "$USIM/tasks/"
```

> ⚠️ 各脚本顶部的 `PROJECT_ROOT = "/home/0000_wy/unitree/unitree_sim_isaaclab"` 是硬编码路径,部署后请改成你的 unitree_sim_isaaclab 实际路径(涉及 `transport_demo.py`、`grasp-expert.py`、`head_cam_calib.py`、`make_yolo_dataset.py` 等)。

## 3. 完整流程与各文件职责(按项目推进顺序)

### 阶段 0:资产与场景搭建

| 文件 | 作用 |
|---|---|
| `measure_asset.py` | 打印候选 USD 工件的尺寸/单位/已有物理 API,为 Dex3 捏合选定缩放 |
| `prepare_drill_asset.py` | YCB 电钻只有视觉网格,补 RigidBodyAPI + 凸分解碰撞体,生成物理可用的新 USD |
| `prepare_factory_asset.py` | Factory 螺栓自带装配关节会把刚体钉回原点,生成禁用全部关节的自由刚体版本 |
| `prepare_table_asset.py` | 从带黄色料框的桌子资产派生无框干净桌面 |
| `tasks/rl/g1_transport/transport_scene_cfg.py` | 双桌搬运场景定义:两张互相垂直的桌子、B 桌放置分区标记、7 个工件排布、机器人初始位姿 |
| `view_transport_scene.py` | 只加载场景跑物理的查看器,布局调整与工件选型都在这里目视确认 |

### 阶段 1:行走策略(浮动基座 RL)

| 文件 | 作用 |
|---|---|
| `tasks/rl/g1_locomotion/locomotion_env_cfg.py` | 平地速度跟踪任务:观测 69 维/动作 19 维(双腿+左臂摆动平衡),**右臂不在动作空间**,为后续 BC 抓取保留 |
| `tasks/rl/g1_locomotion/agents/skrl_ppo_cfg.yaml` | skrl PPO 超参 |
| `inspect_walk_policy.py` | 不启动 Isaac,解析训练出的 checkpoint 格式(skrl/rsl_rl)与网络 I/O 维度,供 transport_demo 的策略适配层使用 |

### 阶段 2:抓取技能(固定基座,专家示教 → 行为克隆)

| 文件 | 作用 |
|---|---|
| `tasks/rl/g1_pickplace/pickplace_env_cfg.py` + `mdp/` | 固定基座抓取环境:28 维关节位置动作(双臂+双手),110 维观测,随机化工件出生点与放置目标 |
| `inspect-robot.py` | 打印全部关节名/限位/默认值与连杆名,确定手指关节闭合方向与末端连杆——专家脚本的全部先验 |
| `grasp-expert.py` | **核心专家脚本**:先规划后执行——HOLD 读工件位姿 → 离线阻尼最小二乘 IK 解出关节路标(控制点为捏合中心)→ 各阶段(UP/TRAV/DESCEND/INSERT/CLOSE/LIFT/CARRY/LOWER/RELEASE/RETREAT)做关节空间慢速插值;`--workpiece {cube,bolt,nut,drill}` 切换工件与抓取档案(侧捏/顶抓);`--save_dataset` 记录成功回合的 (obs, action);`--action_noise` 施加 DART 式执行噪声抗复合误差 |
| `dataset_check.py` | 离线验证数据集确实描述完整抓-放循环 |
| `bc_dataset_merge.py` | 合并干净数据与多档噪声数据为训练用单文件 |
| `bc_train.py` | BC 训练器,网络结构与 skrl PPO 模型完全一致 |
| `bc_play.py` | 在抓取环境回放 BC 策略,报成功率、录像,供人眼检查实际效果 |
| `bc_obs_diff.py` / `bc_rollout_diag.py` | 两个离线诊断器:前者对比两套环境的 110 维观测找出让策略翻车的字段;后者检测"最后动作捷径"式过拟合 |
| `finetune_from_bc.py` / `train-catch.py` / `play-catch.py` / `play_rl.py` | 可选的 PPO 微调链路(BC 权重热启动 + 价值头预热)及其评估回放 |

### 阶段 3:整合搬运演示(本项目核心)

| 文件 | 作用 |
|---|---|
| `transport_demo.py` | **主演示脚本**:行走策略适配层、BC 策略按工件切换、TURN/NAV/ALIGN/STAND/BC/BACKOFF/TRANSIT/FACE 等相位状态机、M1动态对接、M2站立抓放、M3持物搬运、M4多工件循环搬运，阶段任务评分、录像、批量统计、全部视觉方案评估(见阶段 4) |
| `parse_m4_stats.py` / `parse_m4_round3.py` | 批量日志挖掘:把停靠误差、抓取遥测与最终判定(PLACED/DROPPED/…)关联,定位失败工况 |

### 阶段 4:视觉层(标定 → YOLO分类 → 影子测量)

| 文件 | 作用 |
|---|---|
| `head_cam_calib.py` | 第一步标定:头部相机在各站位/俯仰角下对每个工位的可见性、投影-反投影链的几何误差、逐类别"深度表面 vs USD 根原点"偏移。结论:停靠位工件在视场外且修正常数随视角变号 → **定位不交给视觉** |
| `vision_gate.py` | 视觉公共模块:相机内外参与投影数学、`VisionShadow`(影子日志)、`YoloGate`(分类)、`GateViewer`(RGB+深度实时浮窗,cv2/omni.ui 双后端) |
| `make_yolo_dataset.py` | 在仿真里自动生成带标注的 YOLO 训练集:工件位姿抖动、相机视角与灯光域随机化、真值投影 3D 框 + 深度遮挡检查 |
| `make_vision_report.py` | 汇总影子 CSV,输出逐类别误差统计 |

## 4. 许可与致谢

- 本仓库代码:Apache-2.0。
- [unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)(Unitree Robotics,Apache-2.0):G1+Dex3 机器人模型与仿真层,本仓库**不内嵌**该项目,请按 2.2 节单独克隆并叠加 `tasks/rl/` 包。
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)(BSD-3-Clause):仿真与 RL 框架。
- YCB / Factory 资产来自 NVIDIA Isaac Sim 官方资产库。
