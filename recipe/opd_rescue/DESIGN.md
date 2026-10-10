# OPD 竞赛场迁移设计（recipe/opd_rescue）

2026-10-11。本目录是把 OPD 项目（JamesJunyuGuo/OPD，方法实验室）的方法与基线家族迁入 verl
训练场的脚手架。`shaping.py` 是从 OPD `pilot/baselines/shaping.py` 原样搬来的**已核验**
基线数学（每个函数的 docstring 带原始仓库 file:line / 论文公式出处），纯 torch + python、
零依赖，可直接被任何 trainer 引用。

## 0. 底座决定（待定，本分支不替你做）

本 fork 目前是**原版 verl（2025-08 HEAD）**，没有 agent 环境层。两个选项：

- **选项 A（推荐）**：另 fork `ZJU-REAL/SDAR`（verl-agent 血统，2026-10 仍在更新）作为竞赛场
  仓库。理由：`agent_system/environments/env_package/` 自带 alfworld / webshop / search
  （正是预定的三件套主战场），`examples/grpo_opd_trainer`、`grpo_opsd_trainer` 是现成的
  OPD 训练样板，且 SDAR/AgentOPSD/OPID 三个基线可以在同一血统里原样运行。本目录的文件
  原样拷过去即可用。
- **选项 B**：留在本 fork，把 SDAR 的 `agent_system/` + `gigpo/` 作为快照导入（Apache-2.0，
  保留 LICENSE/Notice 并在提交里注明来源 commit）。工程上可行，但要自己背着 verl 核心版本
  差异（本 fork 的 core 比 verl-agent 分叉点新一年）。

## 1. 钩子点地图（来自五个基线仓库的代码级报告，OPD docs/baselines_impl.md 有全表）

verl-agent 血统里，整个基线家族只用三个钩子位，我们的方法恰好也只需要这三个加一个新的：

| 钩子 | 位置（以 SDAR/AgentOPSD fork 为准） | 谁用 |
|---|---|---|
| **教师打分**：同一 actor 对"特权上下文 + 原采样 response"算 log_prob，无梯度 | `rlsd_ray_trainer.py` `build_teacher_batch` + `_compute_teacher_log_probs`（actor_rollout_wg.compute_log_prob） | RLSD/SDAR/AgentOPSD/OPID/Skill-SD，我们的 ρ 诊断与选择算子打分 |
| **优势替换/整形**：在 compute_advantage 之后、update_actor 之前改 `batch.batch["advantages"]` | AgentOPSD `opsd_ray_trainer.py:496`；OPID `core_gigpo.py:471-480`；SOD `ray_trainer.py:878-929` | 全部 adv_shaping 基线；我们的 group_aug（救援成员替入 + 裁剪 IS） |
| **加性损失**：dp_actor.update_policy 里 policy_loss 之外加一项 | SDAR `dp_actor.py:445-458` | SDAR；（我们不用——L2 教训） |
| **救援 rollout 钩子（要新写的唯一一块）**：rollout 循环里对全败组触发带特权上下文的续写 + 放手验证 | `agent_system/multi_turn_rollout/rollout_loop.py`（verl-agent 的逐轮循环，天然支持 per-step prompt 改写——OPID 的 augmented observation 就在这里做的） | 我们的 pivot-handoff / BoN 救援；PSP 复现 |

要点：verl-agent 把**每个环境步作为一个样本行**（traj_uid + turn_step 分组），这与 OPD 的
per-turn example 构造同构——OPD `build_turn_example` 的字节级对齐纪律在这边对应"teacher
prompt 重分词后 response token 原样拼接"（`build_teacher_batch` 已这么做）。

## 2. 迁移清单（冻结配方后执行）

1. 环境协议对齐：ALFWorld prompt 模板、动作语法、历史窗口 5、`enable_thinking=False`
   预填——与 OPD v3 协议逐项对照（OPD `pilot/core/prompts.py` 是权威）；274 评测任务
   zero-overlap 断言照搬。
2. `shaping.py` 接入：adv_shaping 族 = 一个 advantage-postprocess 函数注册点（参照
   AgentOPSD 的 `compute_opsd_turn_advantage` 的插法）；每个基线一个 config 块，超参
   默认值已在 OPD `pilot/baselines/__init__.py` 注册表里。
3. 救援钩子：全败组触发 → 特权上下文续写 ≤N 条 →（可选 j 轮截断 + 无提示放手跑完）→
   环境胜利 + 泄漏过滤 → 按无提示似然 BoN/MH 选 → 作为第 G+1（或替换）成员进组。
   ρ 逐 token 记录，IS 修正用 token 级裁剪（OPD 已修过整条轨迹 exp(−ρ) 的数值爆炸——
   见 OPD commit 8a71be7，不要重犯）。
4. 公平性纪律照搬 OPD `PROGRESS_CN.md` §7.4.1：同起点、同预算记账（环境步）、同精度、
   同评测（temp 0.4 × 3 种子）；跨框架不比绝对数字——pilot 栈的数字只作机制证据，
   主表全部在本框架内重跑。
5. 基线运行顺序（对照 OPD 阶梯）：GRPO → 我们的方法 → PSP → PivotOPD 复现 →
   adv_shaping 族（RLSD/SOD/StepOPSD/AgentOPSD/OPID）→ SDAR → distill 族
   （OPSD/TurnOPD/TCOD）。TurnOPD/TCOD 两处 ⚠️ 细节先对原文核验
   （OPD docs/baselines_impl.md 记录在案）。

## 3. 本目录文件

- `shaping.py` — 已核验的基线数学（与 OPD 同步；改动请两边同步）。
- `DESIGN.md` — 本文件。
- （迁移时新增）`contexts.py` — 特权上下文构造，必须对着本框架的 prompt builder 重写，
  不能从 OPD 直接拷（prompt 格式不同；字节级一致是红线）。
