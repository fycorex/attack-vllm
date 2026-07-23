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

### 3.0 符号、张量形状与“global/local”分别是什么

下表中的 `p` 始终表示**当前这张对抗图唯一使用的代理模型**，即 P1、P2 或 P3 之一；攻击期间不会把两个代理的梯度混合。

| 符号 | 形状 | 代码/数学定义 | 作用 |
| --- | --- | --- | --- |
| `x` 或 `x_clean` | `[1, 3, H, W]` | 原始 source RGB 图，范围 `[0,1]` | 扰动中心与负参考。 |
| `x_adv` | `[1, 3, H, W]` | 优化后的 source 图，满足 `max_abs(x_adv-x_clean) <= epsilon` | 真正提交给黑盒 target 的对抗图。 |
| `E_p^l(x)` | `[1, T_l, d_l]` | 代理 `p` 在视觉层 `l` 输出的**有效 image tokens**；text tokens 不包含在内 | 所有 global/local 表征的来源。 |
| `M_p^l(x)` | `[1, T_l]` Boolean | `E` 中哪些 token 属于当前图像且有效 | 动态分辨率/填充时避免把无效 token 平均进去。 |
| `g_p^l(x)` | `[1, d_l]` | `L2Normalize(masked_mean(E_p^l(x), M_p^l(x)))` | **global 图级特征**：一张图在层 `l` 的单个向量。论文中的 CKA 与 global loss 都使用这种图级几何。省略上标 `l` 的 `g_p(x)` 指最终或接口层的 global 特征。 |
| `e_p^l(x)` | `[1, T_l, d_l]` | 对 `E_p^l(x)` 的每个有效 token 分别 L2 normalize | **local token 特征**：保留 patch/视觉位置级结构，用于局部匹配。 |
| `P` | 13 张图 | 正锚点集合：target 自然图、8 个确定性 target views、4 张同答案同题型图 | 定义“应朝向什么视觉概念”。 |
| `N` | 17 张图 | clean source、8 个 source views、8 个 hard negatives | 定义“应离开什么概念”。 |
| `a` | 一张锚点图 | `a ∈ P` 或 `a ∈ N` | `g_p^l(a)`、`e_p^l(a)` 是攻击前缓存的常量。 |

对每个有效层，global 特征就是：

```text
g_p^l(x) = L2Normalize( sum_r M_r * E_p,r^l(x) / sum_r M_r )
```

而 local 特征是：

```text
e_p,r^l(x) = E_p,r^l(x) / ||E_p,r^l(x)||_2
```

因此，“把对抗图拉近 target、推离 source”并不是直接比较像素：global 部分把 `g_p(x_adv)` 拉向 target 图群的中心、远离 `g_p(x_clean)`；local 部分则让 `e_p(x_adv)` 中的视觉 token 与 target anchors 的 token 产生更高的双向匹配。CKA 只报告前者的**跨图全局几何**，不使用 local matching score。

### 3.1 MaxStrengthHierarchicalDirection 损失

对每个 source-target pair，正锚点有 13 个：自然 target 图像、4 个固定平移视图 `(-4,0),(4,0),(0,-4),(0,4)`、4 个固定缩放视图 `(0.95,0.975,1.025,1.05)`、以及 4 张同标准化答案且同题型的 VQAv2 图像。负锚点为 clean source、它的 8 个确定性视图和 8 张 metadata 选择的 hard negatives。锚点 ID 不使用代理 embedding 或目标攻击结果选择，固定特征在优化前缓存。

同一个代理的文本编码器产生四个 target concept 模板：`a photo of {answer}`、`an image containing {answer}`、`the visual answer is {answer}`、`Question: {question} Answer: {answer}`。其中 `mean(...)` 是集合中向量的算术平均，`Normalize(...)` 是 L2 归一化；`z_pos` 是目标语义端点、`z_neg` 是 source/hard-negative 端点，`d_target` 是希望扰动沿着移动的单位方向：

```text
z_pos    = Normalize(0.5 * mean(target_image_anchor_embeddings)
                   + 0.5 * mean(same_proxy_target_text_embeddings))
z_neg    = Normalize(mean(clean_source_and_hard_negative_embeddings))
d_target = Normalize(z_pos - z_neg)
d_adv    = Normalize(g_p(x_adv) - g_p(x_clean))
```

攻击最小化 `L_direction=1-cos(d_adv,d_target)`，故它要求“从 clean 表征到 adversarial 表征的位移”指向 target direction；`L_endpoint=1-cos(g_p(x_adv),z_pos)` 则直接要求 adversarial 图靠近 target semantic endpoint。层 `l` 使用权重 `(0.10,0.15,0.20,0.25,0.30)` 对应 25%/50%/75%/final/interface；缺失层在可用层间重归一化，Qwen 当前只有 interface。每层 global 项为：

```text
target_centroid_p^l = Normalize( mean_{a in P}( g_p^l(a) ) )
L_global^l          = 1 - cosine( g_p^l(x_adv), target_centroid_p^l )
```

每层 local 项是 `x_adv` 的 token 与 13 个 anchor 的 token **逐 anchor**计算对称 max-cosine 后平均；“逐 anchor”尤其重要，因为 Qwen 的动态分辨率使不同图的 token 数 `T_l` 不同：

```text
M(u, v) = 0.5 * [ mean_r max_s cosine(u_r, v_s)
                + mean_s max_r cosine(u_r, v_s) ]
```

所以 `L_local^l = - mean_{a in P}( M(e_p^l(x_adv), e_p^l(a)) )`；负号表示优化时最大化局部匹配。总损失为：

```text
L = 1.00 * L_direction
  + 0.30 * L_endpoint
  + 0.50 * sum_l(w_l * L_global[l])
  + 0.35 * sum_l(w_l * L_local[l])
  + 0.15 * cosine(g_p(x_adv), g_p(x_clean))
```

最后一项是 `L_src=cos(g_p(x_adv),g_p(x_clean))`；最小化它会使 source global 表征远离。它结合了 UnivIntruder 的同代理文本语义方向、SGHA 的多参考多深度对齐；VEAttack 则作为独立的非定向 image-token 梯度/PNG sanity check。RaPA 5% 可逆视觉 output-projection pruning 已实现为备用分支，但本表结果使用 `rpa_ratio=0`，避免混淆第一批结果。

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

## 6. 文献对比：哪些数字可以比较，哪些不能直接比较

下表优先采用论文官网、CVF Open Access 或 arXiv 原文。`可直接比较`指是否同时满足“单 checkpoint 代理、跨家族 VLM、VQA、同量级 16/255、严格 exact-answer”——没有任何一篇与本实验完全一致，因此论文 ASR 只能作为方法与量级参考，不能当作本实验应达到的硬阈值。

| 工作（状态） | 代理与方法核心 | 目标/任务与关键设置 | 论文报告数字 | 评分口径与和本实验的关系 |
| --- | --- | --- | --- | --- |
| [V-Attack（CVPR 2026；arXiv）](https://arxiv.org/abs/2511.20223) | **单个** CLIP-L/14@336；攻击 attention value features；Self-Value Enhancement + text-guided value manipulation；使用 augmentation | MS-COCO VQA，L∞=16/255，200 steps，crop [0.75,1]；黑盒 LLaVA/InternVL/DeepSeekVL/GPT-4o | 单代理 VQA：LLaVA 54.2%，InternVL 35.2%，DeepSeekVL 24.0%，GPT-4o 39.1% | 最接近我们的“CLIP 单代理→跨家族 VLM”。但 ASR 是 GPT-4o 自动评审三档分数 `1/0.5/0` 的平均，包含部分成功；**不能与 strict exact-answer TASR 直接等同**。 |
| [UnivIntruder（CCS 2025；arXiv）](https://arxiv.org/abs/2505.19840) | **单个公开 CLIP**；目标文本概念驱动 universal targeted perturbation | 100 个 16/255 样本；Claude-3.5、GPT-4、GPT-4o 的图像分类式短提示 | 全 ASR：80% / 64% / 54%；其中 target-only：52% / 34% / 16% | 论文把 target-only 的 `Deception` 和 source+target 的 `Ambiguity` 都计为成功。它支持“一个公开 CLIP 可迁移”的可行性，但任务与评分宽于 VQAv2 exact answer。 |
| [RaPA（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Su_RaPA_Enhancing_Transferable_Targeted_Attacks_via_Random_Parameter_Pruning_CVPR_2026_paper.html) | 同一 surrogate checkpoint 的可逆随机参数剪枝；不需要第二个代理 | CNN→Transformer targeted **分类**迁移 | 困难跨架构设置报告约 33.3% ASR，并较当时基线高至 11.7 pp | 不是 VQA，也不提供 strict-answer 对照；本项目只把 2%/5% visual output-projection pruning 作为 P2/P3 仍为零时的可控消融。 |
| [VEAttack（arXiv:2505.17440）](https://arxiv.org/abs/2505.17440) | 单一 LVLM vision encoder；最小化 clean/adv image-token 相似度 | 下游无关、非定向；包含 VQA 性能下降评估 | 文中报告 VQA 性能下降 75.7% | **不是 targeted ASR**。本项目仅用它验证 token 梯度、预处理和 PNG 序列化路径，不能与 TASR 比。 |
| [SGHA-Attack（arXiv:2602.01574，预印本）](https://arxiv.org/abs/2602.01574) | 多目标参考、多层 global/local 对齐、视觉-文本语义引导 | 黑盒 VLM targeted transfer | 论文称优于既有 targeted baselines；设置与评估集不同 | 本项目保留其多参考和分层对齐思想，但用 VQAv2 真 target 图像与同答案 anchors，不把它的结果与本表做数值比较。 |
| [Omni-Attack（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_Omni-Attack_Adversarial_Attacks_on_Open-Ended_VQA_in_Black-Box_Multimodal_LLMs_CVPR_2026_paper.html) | question-conditioned text/visual target construction；其最佳实践使用多个 CLIP/SigLIP surrogate 与多视图 | AdvRobustBench 开放式 VQA/OCR，ε=8/255 | GPT-4.1 上最高 71.8% targeted ASR | 是强开放 VQA 参考和潜在上界，但**不是纯单代理**，且目标构造、数据与评审协议不同，不能作为单 proxy 基准。 |
| **本实验：MaxStrengthHierarchicalDirection** | 每图严格一个 P1/P2/P3 checkpoint；同代理文本方向 + 13 anchors + 多层 global/local token + EOT | VQAv2；ε=16/255；300 steps；5 restarts；8 EOT；Gemma/InternVL | 已完成 P2/P3：P3→T2 严格 6/14=42.9% | 自动短答案 + VQAv2 标准化 **exact match**；target/clean/random 三个 guard 都通过才算成功，是本表最严格的口径之一。 |

### 6.1 对本 pilot 的合理解读阈值

若最终按原计划使用 12 个独立 test pairs，则单个严格命中对应 8.3 个百分点；当前 P2/P3 的 14 图结果仅是包含 development 的 interim batch，不应用作最终 test-only 结论。

| 严格成功数（12 test pairs） | strict TASR | 适合的解释 |
| ---: | ---: | --- |
| 0/12 | 0.0% | 尚未建立定向迁移。 |
| 1/12 | 8.3% | 候选信号；必须用第二 seed 或独立 pair 复现。 |
| 2/12 | 16.7% | 已建立可信的非零 single-proxy transfer。 |
| 3–4/12 | 25.0–33.3% | 对 strict exact-answer、跨 family 的 pilot 已较强；可与 V-Attack 的 InternVL 量级讨论，但必须注明评分不同。 |
| ≥5/12 | ≥41.7% | 很强；应优先复核 pair screening、clean/random guard、序列化和回答标准化，排除数据泄漏或过宽松匹配。 |

因此本项目的近期目标不是机械追逐“50%”，而是先在 held-out test 上达到至少 2/12 严格命中并以 seed 43 重现；3–4/12 已是有信息量的正向结果。我们当前 P3→T2 的 6/14 是鼓舞信号，但因为混有 dev、还未做 seed 43，不能填入上表的最终 test 判断。

## 7. 当前工程状态与下一步

- **P1/Qwen 攻击正在运行。** 初次运行失败是因为动态分辨率导致不同 anchor 的 image-token 数不同；局部损失已改为逐 anchor 求对齐分数后平均，单样本 Qwen smoke 已通过。P1 的 replay CLI 也已启用。
- 当前 P1 调度使用同一强攻击配置和 10 小时上限，最后两小时自动回放 T1/T2。
- P2/P3 之后需在同一对抗样本上跑 8/255，并对严格成功案例以 seed 43 重现。
- 若 P3→T2 的优势仍保留，应逐图比较 source-target 语义距离、答案类别、局部 token 对齐、global CKA、预处理分辨率、EOT 下的损失稳定性与目标输出稳定性，解释为什么 transferability 不随 global CKA 单调变化。

本报告不会将 P1 未完成、未复现的成功、或弱行为改变写成最终结论。
