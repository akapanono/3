# HDSA-ERC 小步调参改进文档

## 1. 当前阶段判断

当前模型已经不适合继续大改结构。前几轮实验说明：

```text
1. HDSA-ERC 主线是有效的。
2. 域子锚点防塌缩版本是当前最好的基础版本。
3. 一次性加入过多混淆类别优化模块会破坏原本稳定的锚点结构。
```

上一轮防塌缩版本的最好结果达到：

```text
test weighted F1 ≈ 0.7042
same_anchor_sim_mean ≈ 0.47
```

说明：

```text
1. 域子锚点没有整体塌缩。
2. happy、frustrated、neutral 等类别相较早期版本有明显提升。
3. 当前模型已经具备进一步调参的基础。
```

但最新一次全量混淆类别优化实验表现不佳：

```text
test weighted F1 ≈ 0.6797
same_anchor_sim_mean 从 0.94 继续升到 0.98 左右
```

说明：

```text
1. 新增模块过多、过强。
2. 子锚点结构再次出现塌缩趋势。
3. 混淆类别边界并没有真正改善。
4. 当前应该回到最优防塌缩版本，进行小步消融和参数调优。
```

因此，本次改进目标是：

```text
回退到最佳防塌缩版本，
一次只加入一个小改动，
逐步验证哪个参数或模块真正有效。
```

---

## 2. 当前最重要的问题

### 2.1 不要继续堆模块

上一轮一次性加入了：

```text
class-adaptive EMA
EMA fallback
pairwise confusion loss
pairwise anchor loss
happy CE weight
intensity head
```

结果导致：

```text
1. same_anchor_sim_mean 重新升高到 0.98 左右。
2. 域子锚点重新塌缩。
3. happy F1 下降。
4. frustrated / angry 混淆没有改善。
5. weighted F1 明显低于上一轮最佳结果。
```

所以后续不能再“全开模块”。

---

### 2.2 需要保护域子锚点结构

HDSA-ERC 的核心创新是：

```text
每个情感类别由多个域子锚点表示，
这些子锚点用于刻画同一情感类别内部的不同表达子域。
```

因此，每次实验都必须优先观察：

```text
same_anchor_sim_mean
same_anchor_sim_max
```

如果某次改动让：

```text
same_anchor_sim_mean > 0.85
```

尤其是接近：

```text
0.95 ~ 1.00
```

即使 F1 短期没有明显下降，也不建议采用，因为它破坏了模型的核心结构。

---

## 3. 本轮调参总原则

本轮调参遵循四个原则：

```text
1. 以最佳防塌缩版本作为 baseline。
2. 每次只加入一个模块或只调整一组参数。
3. 每次实验都记录 weighted F1、macro F1、same_anchor_sim_mean、混淆矩阵。
4. 只有在不破坏域子锚点结构的情况下，才保留该改动。
```

不要同时启用：

```text
class-adaptive EMA
EMA fallback
pairwise loss
pairwise anchor loss
happy CE weight
intensity head
```

必须逐个测试。

---

## 4. 固定基础版本

请 Codex 先恢复到上一轮防塌缩版本，也就是保留：

```text
1. KMeans 域子锚点初始化
2. L_preserve 聚类中心保持损失
3. 弱化 L_center
4. 增强 L_div
5. OT sharpen
6. 高置信 EMA 更新
7. loss_ce + loss_proto + loss_compact
8. anchor stats / OT stats / EMA counts 日志
```

基础推荐参数：

```bash
--center_weight 0.1
--div_weight 1.0
--preserve_weight 0.5
--same_upper 0.90

--ot_epsilon 0.02
--ot_iters 50
--ot_sharpen_power 2.0

--prototype_momentum 0.95
--ema_conf_threshold 0.45

--proto_loss_weight 0.5
--compact_loss_weight 0.1
--anchor_logit_weight 0.3
```

这一个版本作为后续所有实验的 baseline。

---

## 5. 实验 0：复现防塌缩 baseline

### 5.1 目的

先确认当前代码仍然能复现上一轮最佳水平。

目标：

```text
weighted F1 接近 0.6978 ~ 0.7042
same_anchor_sim_mean 保持在 0.45 ~ 0.65 左右
```

### 5.2 运行配置

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --use_hdsa \
  --dataset_name IEMOCAP \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --domain_anchor_path ./domain_anchors/IEMOCAP_M3_hyp.pt \
  --num_subanchors 3 \
  --anchor_dim 256 \
  --center_weight 0.1 \
  --div_weight 1.0 \
  --preserve_weight 0.5 \
  --same_upper 0.90 \
  --ot_epsilon 0.02 \
  --ot_iters 50 \
  --ot_sharpen_power 2.0 \
  --prototype_momentum 0.95 \
  --ema_conf_threshold 0.45 \
  --proto_loss_weight 0.5 \
  --compact_loss_weight 0.1 \
  --anchor_logit_weight 0.3
```

注意：

```text
本实验不要开启：
class_adaptive_ema
ema_fallback
pair_loss
pair_anchor_loss
happy_ce_weight
intensity_head
```

---

## 6. 实验 1：只测试 class-adaptive EMA

### 6.1 目的

上一轮实验说明：

```text
angry / frustrated 的 ema_update_counts 长期为 0
```

说明这两个类别的高置信样本不足。  
但是全量方案中阈值太低、fallback 太强，导致锚点结构被破坏。

本实验只测试类别自适应 EMA，不开启其他模块。

### 6.2 推荐参数

```bash
--class_adaptive_ema
--low_conf_classes angry,frustrated
--low_conf_threshold 0.40
--happy_conf_threshold 0.42
--default_ema_conf_threshold 0.45
```

说明：

```text
angry/frustrated 阈值从 0.45 降到 0.40，
不要直接降到 0.38，
避免低置信样本过多参与更新。
```

### 6.3 不开启的模块

```text
不要开启 EMA fallback
不要开启 pairwise confusion loss
不要开启 pairwise anchor loss
不要开启 happy CE weight
不要开启 intensity head
```

### 6.4 观察指标

重点看：

```text
1. angry ema_update_counts 是否不再为 [0,0,0]
2. frustrated ema_update_counts 是否不再为 [0,0,0]
3. same_anchor_sim_mean 是否仍低于 0.65
4. angry -> frustrated 是否下降
5. frustrated -> angry 是否下降
6. weighted F1 是否接近或超过 baseline
```

### 6.5 是否保留

保留条件：

```text
weighted F1 不低于 baseline 0.3 个点以上；
same_anchor_sim_mean 不超过 0.70；
angry/frustrated 的 EMA 更新开始生效；
angry/frustrated 互判减少。
```

如果：

```text
same_anchor_sim_mean 重新升到 0.9+
```

则说明阈值仍然过低或 EMA 更新过强，不保留。

---

## 7. 实验 2：只测试轻量 Pairwise Confusion Loss

### 7.1 目的

上一轮全量实验使用：

```text
pair_loss_weight = 0.1
pair_margin = 0.3
```

偏强，可能扰乱了整体决策边界。

本实验只测试一个很弱的 pairwise loss。

### 7.2 推荐参数

```bash
--pair_loss_weight 0.03
--pair_margin 0.20
```

### 7.3 混淆类别对

```python
confusion_pairs = [
    ("happy", "excited"),
    ("angry", "frustrated"),
    ("neutral", "frustrated"),
    ("sad", "frustrated"),
    ("sad", "neutral")
]
```

### 7.4 不开启的模块

```text
不要开启 class-adaptive EMA
不要开启 EMA fallback
不要开启 pairwise anchor loss
不要开启 happy CE weight
不要开启 intensity head
```

### 7.5 观察指标

重点看：

```text
happy -> excited
angry -> frustrated
frustrated -> angry
neutral -> frustrated
frustrated -> neutral
weighted F1
macro F1
same_anchor_sim_mean
```

### 7.6 是否保留

保留条件：

```text
混淆类别错误数下降；
weighted F1 不下降；
same_anchor_sim_mean 不明显升高。
```

如果 pairwise loss 让某一类提升，但整体 F1 下降明显，不保留。

---

## 8. 实验 3：只测试 happy CE 轻量加权

### 8.1 目的

happy 仍然容易被 excited 吞掉。  
但是上一轮使用：

```text
happy_ce_weight = 1.3
```

可能过强，导致 happy / excited 边界不稳定。

本实验只测试轻量 happy 加权。

### 8.2 推荐参数

```bash
--happy_ce_weight 1.10
```

如果已有 class-balanced CE，则在原有权重基础上乘 1.10。

### 8.3 不开启的模块

```text
不要开启 class-adaptive EMA
不要开启 EMA fallback
不要开启 pairwise loss
不要开启 pairwise anchor loss
不要开启 intensity head
```

### 8.4 观察指标

重点看：

```text
happy F1
happy recall
happy -> excited
excited -> happy
excited F1
weighted F1
```

### 8.5 是否保留

保留条件：

```text
happy F1 上升；
happy -> excited 下降；
excited F1 不明显下降；
weighted F1 不下降。
```

如果：

```text
happy F1 上升，但 excited F1 明显下降
```

说明 happy 权重过高，需要改为：

```text
happy_ce_weight = 1.05
```

---

## 9. 实验 4：组合有效模块

只有当实验 1、2、3 中有模块单独有效时，才进行组合实验。

推荐组合顺序：

```text
组合 A：baseline + class-adaptive EMA
组合 B：baseline + pairwise loss
组合 C：baseline + happy_ce_weight
组合 D：baseline + class-adaptive EMA + pairwise loss
组合 E：baseline + class-adaptive EMA + happy_ce_weight
```

不要直接组合：

```text
class-adaptive EMA + pairwise loss + happy CE + intensity head + pair anchor
```

---

## 10. 暂时不建议继续使用的模块

### 10.1 EMA fallback

原因：

```text
fallback soft update 会让低置信样本也参与锚点更新。
上一轮实验中，它可能让锚点更新过于平均，导致 same_anchor_sim_mean 重新升高。
```

暂时关闭：

```bash
# 不使用 --use_ema_fallback
```

后续如果必须使用，只建议：

```text
fallback_momentum = 0.995
只在连续 2 个 epoch 某类 update_count 为 0 时触发
```

### 10.2 Pairwise anchor separation

原因：

```text
它直接作用于锚点中心，可能破坏已经稳定的域子锚点结构。
上一轮实验中 pair_anchor_loss 很快变成 0，但整体结果没有提升。
```

暂时关闭：

```bash
--pair_anchor_loss_weight 0
```

### 10.3 Intensity head

原因：

```text
情绪强度标签是人工弱标签。
happy/sad/frustrated/angry/excited 的强度在真实数据中并不完全固定。
弱标签可能引入噪声，干扰主任务。
```

暂时关闭：

```bash
--intensity_loss_weight 0
```

后续如果重新尝试，建议：

```text
intensity_loss_weight = 0.01 或 0.02
```

而不是 0.05。

---

## 11. 每次实验必须记录的指标

### 11.1 总体指标

```text
accuracy
macro F1
weighted F1
best epoch
dev weighted F1
test weighted F1
```

### 11.2 每类 F1

```text
angry_f1
excited_f1
frustrated_f1
happy_f1
neutral_f1
sad_f1
```

### 11.3 混淆类别错误数量

```text
happy -> excited
excited -> happy
angry -> frustrated
frustrated -> angry
neutral -> frustrated
frustrated -> neutral
sad -> frustrated
sad -> neutral
```

### 11.4 锚点结构指标

```text
same_anchor_sim_mean
same_anchor_sim_max
diff_anchor_sim_mean
diff_anchor_sim_max
```

### 11.5 OT 和 EMA 指标

```text
ot_entropy_mean
ot_max_prob_mean
ema_update_counts
```

---

## 12. 判断一个改动是否有效

一个改动必须同时满足三个条件才算有效：

```text
1. weighted F1 不下降，最好提升。
2. same_anchor_sim_mean 不回到 0.9 以上。
3. 目标混淆类别错误数确实下降。
```

如果只满足其中一个，不建议保留。

例如：

```text
happy F1 上升，但 weighted F1 下降：
    不直接保留，需要降低权重。

angry/frustrated EMA 更新变多，但 F1 下降：
    不直接保留，需要提高阈值。

pairwise loss 降低某个混淆，但 same_anchor_sim_mean 升到 0.95：
    不保留，因为破坏了域子锚点结构。
```

---

## 13. 推荐实验表格

| 实验编号 | 改动 | 关键参数 | 是否开启其他模块 | 目标 |
|---|---|---|---|---|
| Exp0 | baseline 防塌缩版本 | 原最佳参数 | 否 | 复现 0.697~0.704 |
| Exp1 | class-adaptive EMA | low_conf_threshold=0.40 | 否 | 修复 angry/frustrated 不更新 |
| Exp2 | weak pairwise loss | weight=0.03, margin=0.20 | 否 | 降低易混淆互判 |
| Exp3 | happy CE 加权 | happy_weight=1.10 | 否 | 提升 happy |
| Exp4 | Exp1 + Exp2 | 小权重组合 | 否 | 同时改善 EMA 和混淆 |
| Exp5 | Exp1 + Exp3 | 小权重组合 | 否 | 改善 happy 和 EMA |

---

## 14. Codex 实现要求

请 Codex 按以下要求调整代码：

```text
1. 不要再新增大模块。
2. 保持当前 HDSA 防塌缩版本作为基础。
3. 将 class-adaptive EMA、pairwise loss、happy CE weight 都做成可开关参数。
4. 默认全部关闭，避免影响 baseline。
5. 每次只通过命令行参数开启一个模块。
6. 继续输出 anchor stats、OT stats、EMA counts。
7. 新增混淆类别错误数量日志。
8. 新增实验结果保存文件，每个实验自动保存 JSON。
```

---

## 15. 下一轮最推荐先跑的命令

### 15.1 先跑 Exp0

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --use_hdsa \
  --dataset_name IEMOCAP \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --domain_anchor_path ./domain_anchors/IEMOCAP_M3_hyp.pt \
  --num_subanchors 3 \
  --anchor_dim 256 \
  --center_weight 0.1 \
  --div_weight 1.0 \
  --preserve_weight 0.5 \
  --same_upper 0.90 \
  --ot_epsilon 0.02 \
  --ot_iters 50 \
  --ot_sharpen_power 2.0 \
  --prototype_momentum 0.95 \
  --ema_conf_threshold 0.45 \
  --proto_loss_weight 0.5 \
  --compact_loss_weight 0.1 \
  --anchor_logit_weight 0.3
```

### 15.2 再跑 Exp1

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --use_hdsa \
  --dataset_name IEMOCAP \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --domain_anchor_path ./domain_anchors/IEMOCAP_M3_hyp.pt \
  --num_subanchors 3 \
  --anchor_dim 256 \
  --center_weight 0.1 \
  --div_weight 1.0 \
  --preserve_weight 0.5 \
  --same_upper 0.90 \
  --ot_epsilon 0.02 \
  --ot_iters 50 \
  --ot_sharpen_power 2.0 \
  --prototype_momentum 0.95 \
  --ema_conf_threshold 0.45 \
  --proto_loss_weight 0.5 \
  --compact_loss_weight 0.1 \
  --anchor_logit_weight 0.3 \
  --class_adaptive_ema \
  --low_conf_classes angry,frustrated \
  --low_conf_threshold 0.40 \
  --happy_conf_threshold 0.42 \
  --default_ema_conf_threshold 0.45
```

---

## 16. 论文描述可用文字

```text
在进一步实验中，我们发现一次性加入多种混淆类别优化策略会对域子锚点空间造成较强扰动，导致同类子锚点相似度重新升高并削弱模型整体性能。因此，后续优化采用小步消融策略：首先固定已经验证有效的域子锚点防塌缩结构，然后分别考察类别自适应 EMA 阈值、轻量级混淆类别 pairwise loss 以及少数类交叉熵加权对模型的影响。每次实验仅引入一个变量，并同时观察 weighted F1、macro F1、同类子锚点相似度以及关键混淆类别错误数。通过这种方式，可以在保持域子锚点结构稳定的前提下，逐步筛选真正有效的混淆类别优化策略。
```

---

## 17. 本次改进总结

本次不是新增复杂模型，而是进入参数消融阶段。

核心策略：

```text
回退到最佳防塌缩版本；
一次只调一个模块；
先保护域子锚点结构；
再慢慢优化 happy/excited、angry/frustrated、neutral/frustrated 混淆。
```

当前最优先测试：

```text
1. class-adaptive EMA，但不要开 fallback。
2. very weak pairwise loss。
3. happy_ce_weight = 1.10。
```

暂时不要使用：

```text
1. EMA fallback soft update。
2. pairwise anchor separation。
3. intensity head。
```

目标：

```text
稳定复现 weighted F1 ≈ 0.704，
在不破坏 same_anchor_sim_mean 的前提下，
逐步降低几个关键混淆类别的错误数。
```
