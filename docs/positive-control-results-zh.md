# 单代理跨家族迁移攻击：当前实验结果报告

> 结果快照：2026-07-23 05:28 UTC。此文档报告已经完成并回放的样本；它是初步正向对照，不是统计定论。

## 1. 实验问题与口径

问题是：在**每张图只用一个公开代理 checkpoint**、不使用目标梯度、且不以目标输出选择 restart 的条件下，是否能对跨家族 VQA 目标产生定向迁移？代理为 CLIP ViT-L/14（P2）、SigLIP2-So400m（P3）及正在补跑的 Qwen3.5-4B（P1）；目标为 Gemma 4 E2B（T1）和 InternVL3.5-2B（T2）。

主攻击为 `MaxStrengthHierarchicalDirection`：16/255 L∞、300 steps、步长 1/255、momentum=1、5 restarts、8 个可微 EOT 视图；损失结合同代理文本语义方向、13 个目标图像锚点、分层 global/local token 对齐和 source repulsion。详细方法、论文链接和实现边界见 [方法说明](max-strength-positive-control.md)。

“严格 targeted success”同时要求：

1. 目标模型在自然 target image 上答对；
2. clean source 与 matched random-noise 均不输出目标答案；
3. adversarial source 的标准化短答案等于目标答案。

另外记录两种**不能替代 TASR**的诊断：`受控答案改变`（target 正确、clean/random 一致、adversarial 改变）和 `原始输出改变`（target 正确、adversarial 与 clean 不同）。

## 2. 表征提取与 global CKA：具体怎么算

### 2.1 使用的 image-token 表征

CKA 只比较图像表征，不使用问题文本、生成答案或目标梯度。对同一张 RGB 图像 `x`，每个模型取得图像 token `E_m(x)∈R^{T×d}` 和有效 token mask `M`：P2 使用 CLIP 最终视觉 patch tokens（移除 CLS）；P3 使用 SigLIP2 最终有效 patch tokens（移除 special/pool token）；P1 使用 Qwen visual merger 输出、进入语言模型融合前的 image tokens；T1 使用 Gemma 视觉—语言接口 tokens；T2 使用 InternVL connector 输出、插入 LLM 前的 image tokens。若 tap 不能验证为纯 image token，实验应失败而不是静默使用 hidden state 或 logits。

每图先做 masked mean pooling 并 L2 归一化：

```text
z_m(x_i) = L2Normalize( sum_r M_ir * E_m,r(x_i) / sum_r M_ir )
```

因此当前报告的 CKA 是**图级 global CKA**：它不要求两个模型 embedding 维度相同，也不逐 patch 对齐，而是比较同一批图像在两个模型空间中的相对几何结构。

### 2.2 CKA 的实际计算

对同一顺序的 256 张 gallery 图像，堆叠为 `Z_m∈R^{256×d_m}`，计算：

```text
K_m   = Z_m @ Z_m.T
H     = I - (1/n) * 1 @ 1.T
K_m^c = H @ K_m @ H

CKA(p, t) = frobenius_inner(K_p^c, K_t^c)
            / (frobenius_norm(K_p^c) * frobenius_norm(K_t^c))
```

实现细节是：特征以 FP32 保存；CKA 在 CPU FP64 中计算；分母最小截断为 `1e-12`；严格验证 gallery image ID 与顺序完全相同；每个模型对做 100 次按图像重采样 bootstrap，报告 95% percentile 区间。seed 42 的 256 图 gallery 是主 selector 候选，独立的 seed 43 gallery 只检验排名稳定性。两套 gallery 的代理排序 Kendall `τ=1.0`。

这解释了为什么 CKA 是一个数字：它概括了“模型 A 和 B 是否把整批图组织成相似的相似度关系矩阵”。它会随 gallery 数据分布而变化；它不是局部 token 邻域相似度。局部 token 对齐在下面的攻击损失中单独计算。

## 3. 攻击方法与全部关键参数

### 3.1 MaxStrengthHierarchicalDirection 损失

对每个 source-target pair，正锚点有 13 个：自然 target 图像、4 个固定平移视图 `(-4,0),(4,0),(0,-4),(0,4)`、4 个固定缩放视图 `(0.95,0.975,1.025,1.05)`、以及 4 张同标准化答案且同题型的 VQAv2 图像。负锚点为 clean source、它的 8 个确定性视图和 8 张 metadata 选择的 hard negatives。锚点 ID 不使用代理 embedding 或目标攻击结果选择，固定特征在优化前缓存。

同一个代理的文本编码器产生四个 target concept 模板：`a photo of {answer}`、`an image containing {answer}`、`the visual answer is {answer}`、`Question: {question} Answer: {answer}`。目标语义方向为：

```text
z_pos    = Normalize(0.5 * mean(target_image_anchor_embeddings)
                   + 0.5 * mean(same_proxy_target_text_embeddings))
z_neg    = Normalize(mean(clean_source_and_hard_negative_embeddings))
d_target = Normalize(z_pos - z_neg)
d_adv    = Normalize(g_p(x_adv) - g_p(x_clean))
```

攻击最小化 `L_direction=1-cos(d_adv,d_target)`、`L_endpoint=1-cos(g_p(x_adv),z_pos)`，并在可用的 25%/50%/75%/final/interface 视觉层使用权重 `(0.10,0.15,0.20,0.25,0.30)`。某层不存在则在可用层间重归一化；Qwen 当前只有 interface 层。每层的 global 项为 `1-cos(g_adv^l,target_centroid^l)`；local 项是 adversarial token 与 13 个 anchor token 分别计算对称 max-cosine 匹配再平均：

```text
M(u, v) = 0.5 * [ mean_r max_s cosine(u_r, v_s)
                + mean_s max_r cosine(u_r, v_s) ]
```

总损失为：

```text
L = 1.00 * L_direction
  + 0.30 * L_endpoint
  + 0.50 * sum_l(w_l * L_global[l])
  + 0.35 * sum_l(w_l * L_local[l])
  + 0.15 * cosine(g_p(x_adv), g_p(x_clean))
```

最后一项使 source 表征远离。它结合了 UnivIntruder 的同代理文本语义方向、SGHA 的多参考多深度对齐；VEAttack 则作为独立的非定向 image-token 梯度/PNG sanity check。RaPA 5% 可逆视觉 output-projection pruning 已实现为备用分支，但本表结果使用 `rpa_ratio=0`，避免混淆第一批结果。

### 3.2 优化、序列化与数据筛选

主配置：`epsilon=16/255`、300 steps、step size `1/255`、momentum=1.0、5 random restarts、每 step 8 个 EOT 视图；EOT 使用可微 reflection-pad 平移 ±8 px、缩放 `U[0.90,1.10]` 与随机 crop/pad，不用水平翻转、高斯/JPEG、drop path 或 target gradients。每个 restart 从 `x+U[-epsilon,epsilon]` 开始；对 8 个 EOT 梯度平均，按 MI-FGSM 的最小化方向更新并投影回 `[0,1]∩B_∞(x,epsilon)`。最终 restart 只按**PNG 重载后代理损失最低**选择。

每个输出以无损 8-bit PNG 保存、重新读取，并记录 float/PNG 的 L∞、SHA-256、每 step loss、方向对齐和运行时间。当前 P2/P3 结果的 PNG L∞ 均通过 16/255 预算检查。VQAv2 pair 限制为视觉明确的 object/attribute/action、1–3 token 答案；自然 target 对两个目标正确，clean/random 都不应已有目标答案。每对先以一个问题评估；同一 adversarial 图可再回放该 target image 的第二个原始问题。

## 4. 已完成的 16/255 回放结果

每个 P2/P3 单元均已回放 14 张符合筛选条件的图像（5 dev + 9 test）；表中的百分比是 `hit_count / 14`。

| 单代理 → 黑盒目标 | 严格 targeted transfer | 受控答案改变 | 原始输出改变 | 当前解读 |
| --- | ---: | ---: | ---: | --- |
| P2 CLIP → T1 Gemma | 0/14（0.0%） | 5/14（35.7%） | 9/14（64.3%） | 有行为扰动，未观察到严格定向命中。 |
| P2 CLIP → T2 InternVL | 1/14（7.1%） | 5/14（35.7%） | 10/14（71.4%） | 存在单个严格命中，需 seed 43 复现。 |
| P3 SigLIP2 → T1 Gemma | 2/14（14.3%） | 6/14（42.9%） | 10/14（71.4%） | 已有非零严格迁移，但样本数仍小。 |
| **P3 SigLIP2 → T2 InternVL** | **6/14（42.9%）** | 6/14（42.9%） | 11/14（78.6%） | 当前最强、可复现前最值得优先诊断的迁移单元。 |

因此，目前可以诚实地说：**单代理定向迁移的正向对照已经出现**，至少在 P3 SigLIP2→T2 InternVL 的 14 张初步样本上为 6/14；但还不能声称这是总体 ASR，也不能将原始输出改变率当成严格 targeted transfer。

机器可读汇总位于：`outputs/proxy_selector_pilot/positive_control/summaries/transfer_rates.json`；逐图的攻击轨迹、PNG L∞、anchors、方向对齐与耗时位于各自的 `metrics.json`；四种图像条件的原始/标准化 VQA 输出在 `positive_control/vllm/`。

## 5. 已有 CKA 与攻击结果的关系

两组独立 256-image gallery 均得到相同的 CKA 代理排名：对 T1、T2 都是 **P1 Qwen > P3 SigLIP2 > P2 CLIP**，Kendall τ=1。seed 42 / seed 43 的主要 CKA 如下：

| 代理 → 目标 | CKA seed 42 | CKA seed 43 | 当前攻击结果 |
| --- | ---: | ---: | --- |
| P1 → T1 | 0.839 | 0.830 | P1 尚在补跑。 |
| P1 → T2 | 0.434 | 0.364 | P1 尚在补跑。 |
| P3 → T1 | 0.696 | 0.687 | 严格 2/14。 |
| P3 → T2 | 0.367 | 0.303 | 严格 6/14。 |
| P2 → T1 | 0.610 | 0.596 | 严格 0/14。 |
| P2 → T2 | 0.349 | 0.289 | 严格 1/14。 |

一个值得检验的**初步现象**是：在 T2 上，P3 的 CKA 低于 P1，却暂时有最高的观察到的严格 TASR。这既不能证明 CKA 无效，也不能证明 P3 必然最好：P1 尚未完成、每个单元仅 14 张、且攻击损失与代理的文本/局部 token 几何都可能是重要混杂因素。下一步应固定攻击方法、补齐 P1、扩至 20 对/代理并做 seed 43，再计算选择 regret。

CKA 原始文件：`cka/cka_seed42.csv`、`cka/cka_seed43.csv`、`cka/cka_bootstrap.json`。

## 6. 当前工程状态与下一步

- **P1/Qwen 攻击正在运行。** 初次运行失败是因为动态分辨率导致不同 anchor 的 image-token 数不同；局部损失已改为逐 anchor 求对齐分数后平均，单样本 Qwen smoke 已通过。P1 的 replay CLI 也已启用。
- 当前 P1 调度使用同一强攻击配置和 10 小时上限，最后两小时自动回放 T1/T2。
- P2/P3 之后需在同一对抗样本上跑 8/255，并对严格成功案例以 seed 43 重现。
- 若 P3→T2 的优势仍保留，应逐图比较 source-target 语义距离、答案类别、局部 token 对齐、global CKA、预处理分辨率、EOT 下的损失稳定性与目标输出稳定性，解释为什么 transferability 不随 global CKA 单调变化。

本报告不会将 P1 未完成、未复现的成功、或弱行为改变写成最终结论。
