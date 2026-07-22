# 单代理跨家族迁移攻击：正向对照实验

## 目标与边界

本阶段先回答一个更基础的问题：**只用一个公开代理模型生成的扰动，能否让未参与优化的 Gemma 4 E2B 或 InternVL3.5-2B 在 VQA 中发生可复现的目标答案迁移？** 只有先证明攻击存在非零迁移，才讨论 CKA 能否选择最佳代理；CKA 不是攻击损失，也不替代强攻击。

每张对抗图像严格只用一个 checkpoint 反传：CLIP ViT-L/14（P2）、SigLIP2-So400m（P3）或 Qwen3.5-4B（P1）。目标模型只用于最终回放，不提供梯度、不参与 restart 选择。

## 方法：MaxStrengthHierarchicalDirection

这是一种**论文启发的组合正向对照**，不是对任一论文的逐行复现。它保留每项来源工作最关键的可检验机制：

| 来源与可信度 | 本实验保留的机制 | 代码中的具体做法 |
| --- | --- | --- |
| [VEAttack（arXiv:2505.17440；2026-02 修订，官方代码）](https://arxiv.org/abs/2505.17440) | 仅攻击视觉编码器的 image tokens；使干净/对抗视觉 token 去相似 | 先最小化 `L_VE=mean cosine(e(x_adv), e(x_clean))`；这是**非定向**梯度、预处理和 PNG 回放 sanity check，不能当作目标攻击成功。 |
| [UnivIntruder（CCS 2025，arXiv:2505.19840）](https://arxiv.org/abs/2505.19840) | 单个公开 CLIP 代理与文本概念可产生定向迁移 | 用**同一代理**的文本编码器，对 `a photo of {answer}`、`Question: ... Answer: ...` 等模板编码；与目标图像锚点合成语义方向，最小化方向误差和目标端点误差。 |
| [SGHA-Attack（arXiv:2602.01574，预印本）](https://arxiv.org/abs/2602.01574) | 多目标参考、全局/局部 token、多深度对齐 | 13 个正锚点（目标图、8 个确定性视图、4 张同答案同题型图）；在 25%/50%/75%/最终/interface 层对齐全局和局部 token。没有的层自动重归一化权重。 |
| [RaPA（CVPR 2026，官方论文与代码）](https://openaccess.thecvf.com/content/CVPR2026/html/Su_RaPA_Enhancing_Transferable_Targeted_Attacks_via_Random_Parameter_Pruning_CVPR_2026_paper.html) | 在优化中对同一个代理做可逆参数随机化，减少对少数代理参数过拟合 | 默认先不启用；仍为零时只对视觉 attention/MLP 的 output projection 施加 5% 随机 mask，不剪 norm、patch/position embedding、projector 或语言模型；每次前向后断言恢复。 |

总损失为：

`L = 1.00 L_direction + 0.30 L_endpoint + 0.50 Σw_l L_global^l + 0.35 Σw_l L_local^l + 0.15 cos(g_adv, g_clean)`，最小化它。

其中 `L_direction` 将“从 source 表征出发的位移”拉向由**目标图像锚点 + 同代理文本概念**组成的方向；`L_global/local` 同时约束图级语义与 patch/token 局部对应。所有固定 source/target/anchor 特征在攻击前缓存；每次 restart 仅按 PNG 重载后的**代理损失**选择，从不根据目标输出挑选。

## 强攻击配置与数据

- 优先顺序：P2 CLIP → P3 SigLIP2 → P1 Qwen；每次仍只加载一个代理。
- 主设置：`epsilon=16/255`、300 steps、步长 `1/255`、momentum=1、5 restarts、每步 8 个可微 EOT 视图、平移 ±8 px、缩放 U[0.90,1.10]、无翻转/JPEG/高斯噪声；最终为无损 8-bit PNG，并重新读取检查 L∞。
- 数据先选 VQAv2 中视觉明确的 object、attribute（颜色/材质/形状/状态）与 action；答案为 1–3 token。自然目标图上两个目标都答对；clean source 与 matched random-noise 都不能已输出目标答案。可行性阶段每对先用一个问题；成功后再回放同图的第二个原始问题。
- 严格成功：自然目标正确、clean/random 均非目标答案、adversarial 的标准化答案等于目标答案；至少同一目标上两张不同图成功，或两个目标各一张成功，并至少一例以 seed 43 重现。较弱的“受控行为改变”和“原始输出改变”也保存，但绝不混写为严格 TASR。

## 10 小时自动运行：会得到什么

```bash
bash scripts/run_positive_control_10h.sh 10
```

脚本前 8 小时以三个独立 worker 并行生成 16/255 对抗样本，最后 2 小时顺序启动 Gemma/InternVL 回放与汇总；可从已有 `metrics.json` 断点续跑，硬截止记录在 `outputs/proxy_selector_pilot/ten_hour_run/status.json`。以当前 CLIP/SigLIP2 实测速度，**预期**得到约 20–25 张已回放图像（不是“每个 proxy 20 张”的保证）；Qwen 较慢，实际数目取决于模型分配、重启数及 vLLM 延迟。

10 小时后应有：

- 每个代理×目标的 `hit_count / clean_valid_count`、严格 TASR、random TASR、`DeltaTASR`、受控答案改变率、原始输出改变率；
- 每张图的 loss trajectory、梯度/方向/逐层 global-local 对齐前后值、float 与 PNG L∞、代理/目标预处理分辨率、anchors、seed、耗时；
- clean / random / adversarial / natural-target 的原始与标准化 VQA 输出，避免把模型本身的回答不稳定性误判为迁移；
- 既有 CKA（两组 256 图 gallery、bootstrap 区间、排名稳定性）作为后续 selector 分析的协变量，而非本轮攻击的判据。

关键文件：`positive_control/summaries/transfer_rates.json`、每图 `metrics.json`、`positive_control/vllm/`、`cka/cka_seed42.csv`、`cka/cka_seed43.csv` 与 `cka/cka_bootstrap.json`。

## 初步结果如何解释、后续如何推进

目前已验证 VEAttack 的代理损失明显下降，说明图像梯度、token 路径和 PNG 预算路径可用；但这**不等价于**定向迁移。此前少量严格回放尚未建立可复现的非零 TASR，因此不应宣称单代理定向迁移已成功。若 16/255 首轮出现非零严格 hit，先冻结方法、扩展到 20 对/代理并以 seed 43 复现，再在同一批样本运行 8/255；若全为零，则输出完整诊断（VE 改变率、方向/分层对齐、梯度零比例、四类输出、L∞ 与版本），停止无限扩参，之后再决定是否加入 value-feature fallback 或改变问题/目标选择。
