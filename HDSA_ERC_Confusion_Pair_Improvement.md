# HDSA-ERC 混淆类别定向改进文档

## 1. 当前实验结论

当前 HDSA-ERC 在加入域子锚点防塌缩机制后，整体指标已经明显提升：

```text
accuracy = 0.6973
macro F1 = 0.6866
weighted F1 = 0.6978
```

训练过程中最好结果出现在 epoch 7：

```text
test weighted F1 = 0.7042
test macro F1 = 0.6904
```

这说明当前主模型方向是有效的，尤其是域子锚点塌缩问题已经明显缓解。  
从日志看：

```text
same_anchor_sim_mean ≈ 0.47
```

相比之前接近 1.0 的情况，说明同类子锚点已经不再整体塌缩。

但是从最新混淆矩阵看，模型仍然存在几组主要混淆：

```text
1. happy ↔ excited
2. angry ↔ frustrated
3. neutral ↔ frustrated
4. sad ↔ frustrated / neutral
```

本次改进目标是：在不破坏当前 HDSA 主结构和防塌缩效果的前提下，针对这些混淆类别进行定向优化。

---

## 2. 当前主要混淆问题

### 2.1 happy 被 excited 吞掉

最新混淆矩阵中：

```text
happy -> excited = 41
happy -> happy = 80
```

当前 happy 指标：

```text
precision = 0.5556
recall = 0.5594
F1 = 0.5575
support = 143
```

虽然 happy 已经比上一轮明显提升，但它仍然容易被 excited 吞掉。  
原因是 happy 和 excited 都是正向情绪，语义接近，而且 happy 样本数较少。

---

### 2.2 angry 与 frustrated 混淆严重

最新混淆矩阵中：

```text
angry -> frustrated = 43
frustrated -> angry = 56
```

当前 angry：

```text
F1 = 0.6667
```

当前 frustrated：

```text
F1 = 0.6693
```

angry 和 frustrated 都是负向高唤醒情绪，经常共享抱怨、反驳、不满等表达，因此边界比较模糊。

另外，训练日志显示：

```text
angry ema_update_counts = [0, 0, 0]
frustrated ema_update_counts = [0, 0, 0]
```

说明这两个类别的域子锚点没有获得高置信 EMA 更新，这可能是 angry/frustrated 边界仍不清晰的重要原因。

---

### 2.3 neutral 与 frustrated 混淆仍然存在

最新混淆矩阵中：

```text
neutral -> frustrated = 53
frustrated -> neutral = 50
```

neutral 和 frustrated 的混淆主要是情绪强度边界问题。  
neutral 通常是低情绪强度，frustrated 是负向但强度更高。  
如果模型对情绪强度不敏感，就容易把轻微 frustrated 判为 neutral，或者把带有负面语气的 neutral 判为 frustrated。

---

### 2.4 sad 与 frustrated / neutral 混淆

最新混淆矩阵中：

```text
sad -> frustrated = 31
sad -> neutral = 23
```

sad 整体 F1 仍然较高：

```text
F1 = 0.7950
```

但 sad 仍有部分被分到 frustrated 和 neutral，说明模型对负向低唤醒和负向高唤醒边界仍有提升空间。

---

## 3. 本次改进目标

本次改进重点是混淆类别定向优化：

```text
1. 降低 happy -> excited 的误判。
2. 降低 angry -> frustrated 和 frustrated -> angry 的互判。
3. 降低 neutral -> frustrated 和 frustrated -> neutral 的互判。
4. 让 angry 和 frustrated 的域子锚点能够参与 EMA 更新。
5. 在损失函数中显式增强易混淆类别之间的边界。
6. 增加日志，单独观察混淆类别对的错误数量。
```

---

## 4. 改进一：类别自适应 EMA 阈值

### 4.1 问题

当前 EMA 使用统一阈值，例如：

```text
ema_conf_threshold = 0.45
```

但 angry 和 frustrated 的 EMA 更新数始终为 0：

```text
angry = [0, 0, 0]
frustrated = [0, 0, 0]
```

说明这些类别的 OT 分配置信度不够高，导致其域子锚点长期不更新。

### 4.2 改进

使用类别自适应阈值：

```text
angry:      0.38
frustrated: 0.38
happy:      0.42
excited:    0.45
neutral:    0.45
sad:        0.45
```

### 4.3 新增参数

```python
parser.add_argument("--class_adaptive_ema", action="store_true")
parser.add_argument("--default_ema_conf_threshold", type=float, default=0.45)
parser.add_argument("--low_conf_classes", type=str, default="angry,frustrated")
parser.add_argument("--low_conf_threshold", type=float, default=0.38)
parser.add_argument("--happy_conf_threshold", type=float, default=0.42)
```

### 4.4 代码示例

```python
def build_class_thresholds(label2id, args):
    thresholds = {
        cid: args.default_ema_conf_threshold
        for cid in range(len(label2id))
    }

    for name in args.low_conf_classes.split(","):
        name = name.strip()
        if name in label2id:
            thresholds[label2id[name]] = args.low_conf_threshold

    if "happy" in label2id:
        thresholds[label2id["happy"]] = args.happy_conf_threshold

    return thresholds
```

EMA 更新时：

```python
threshold = class_thresholds[c]
keep = max_prob >= threshold
```

---

## 5. 改进二：EMA fallback soft update

### 5.1 问题

即使降低阈值，某些类别仍可能一个 epoch 没有高置信样本。  
如果完全不更新，这些类别的域子锚点会长期静止。

### 5.2 改进

如果某个类别高置信更新数为 0，则使用 soft assignment 做一次低强度 EMA 更新。

推荐：

```text
fallback_momentum = 0.98
```

### 5.3 新增参数

```python
parser.add_argument("--use_ema_fallback", action="store_true")
parser.add_argument("--fallback_momentum", type=float, default=0.98)
```

### 5.4 代码示例

```python
if update_counts[c].sum() == 0 and args.use_ema_fallback:
    z_c = reps[idx]
    gamma_c = soft_targets[idx, c]  # [N_c, M]

    for m in range(M):
        weight = gamma_c[:, m]
        mass = weight.sum()
        if mass.item() <= 1e-6:
            continue

        mean = (weight[:, None] * z_c).sum(dim=0) / mass
        mean = F.normalize(mean, dim=-1)

        old = model.domain_anchors[c, m]
        new = args.fallback_momentum * old + (1 - args.fallback_momentum) * mean
        model.domain_anchors[c, m] = F.normalize(new, dim=-1)
```

---

## 6. 改进三：Pairwise Confusion Loss

### 6.1 设计动机

普通 CE 是全局分类损失，不会重点关注：

```text
happy vs excited
angry vs frustrated
neutral vs frustrated
```

因此加入样本级 pairwise margin loss，专门惩罚易混淆类别对。

### 6.2 混淆类别对

```python
confusion_pairs = [
    ("happy", "excited"),
    ("angry", "frustrated"),
    ("neutral", "frustrated"),
    ("sad", "frustrated"),
    ("sad", "neutral")
]
```

### 6.3 损失定义

如果真实类别是 `a`，其混淆类别是 `b`，要求：

```text
logit_a > logit_b + margin
```

损失：

```text
L_pair = mean ReLU(margin - (logit_true - logit_confused))
```

推荐：

```text
pair_margin = 0.3
pair_loss_weight = 0.1
```

### 6.4 代码

```python
def pairwise_confusion_loss(logits, labels, label2id, pairs, margin=0.3):
    losses = []

    for a, b in pairs:
        if a not in label2id or b not in label2id:
            continue

        ia = label2id[a]
        ib = label2id[b]

        mask_a = labels == ia
        if mask_a.any():
            diff = logits[mask_a, ia] - logits[mask_a, ib]
            losses.append(F.relu(margin - diff).mean())

        mask_b = labels == ib
        if mask_b.any():
            diff = logits[mask_b, ib] - logits[mask_b, ia]
            losses.append(F.relu(margin - diff).mean())

    if len(losses) == 0:
        return logits.new_tensor(0.0)

    return torch.stack(losses).mean()
```

---

## 7. 改进四：混淆类别锚点中心分离损失

### 7.1 设计动机

Pairwise Confusion Loss 约束 logits。  
为了让原型空间也更清晰，可以对混淆类别的锚点中心加入弱分离约束。

### 7.2 损失定义

每个类别的锚点中心：

```text
μ_c = mean_m a_{c,m}
```

对于混淆类别对 `(a,b)`：

```text
L_pair_anchor = mean ReLU(cos(μ_a, μ_b) - upper)
```

推荐：

```text
pair_anchor_upper = 0.20
pair_anchor_loss_weight = 0.05
```

### 7.3 代码

```python
def pairwise_anchor_separation_loss(anchors, label2id, pairs, upper=0.20):
    anchors = F.normalize(anchors, dim=-1)
    centers = F.normalize(anchors.mean(dim=1), dim=-1)

    losses = []

    for a, b in pairs:
        if a not in label2id or b not in label2id:
            continue

        ia = label2id[a]
        ib = label2id[b]

        sim = torch.sum(centers[ia] * centers[ib])
        losses.append(F.relu(sim - upper))

    if len(losses) == 0:
        return anchors.new_tensor(0.0)

    return torch.stack(losses).mean()
```

注意：该 loss 权重不能太大，否则可能破坏情感类别之间的自然语义关系。

---

## 8. 改进五：happy 类 CE 加权

### 8.1 设计动机

happy 样本数较少，而且容易被 excited 吞掉。  
可以适当提高 happy 类 CE 权重。

推荐：

```text
happy_ce_weight = 1.3
```

### 8.2 新增参数

```python
parser.add_argument("--happy_ce_weight", type=float, default=1.3)
```

### 8.3 代码

```python
ce_weights = torch.ones(num_classes)

if "happy" in label2id:
    ce_weights[label2id["happy"]] = args.happy_ce_weight

loss_ce = F.cross_entropy(logits, labels, weight=ce_weights.to(logits.device))
```

如果已经启用 class-balanced CE，则在原权重基础上乘：

```python
ce_weights[label2id["happy"]] *= args.happy_ce_weight
```

---

## 9. 改进六：情绪强度辅助头

### 9.1 设计动机

neutral/frustrated 和 happy/excited 的混淆，与情绪强度有关。  
引入一个轻量情绪强度辅助头，帮助模型学习情绪强弱边界。

### 9.2 强度标签

不需要额外标注，直接由情感类别构造弱标签：

```python
intensity_map = {
    "neutral": 0,
    "sad": 1,
    "happy": 1,
    "frustrated": 2,
    "angry": 2,
    "excited": 2
}
```

其中：

```text
0 = low intensity
1 = medium intensity
2 = high intensity
```

### 9.3 模型结构

新增：

```python
self.intensity_classifier = nn.Linear(anchor_dim, 3)
```

forward 输出：

```python
logits_intensity = self.intensity_classifier(z)
```

### 9.4 损失

```python
loss_intensity = F.cross_entropy(logits_intensity, intensity_labels)
```

推荐：

```text
intensity_loss_weight = 0.05
```

---

## 10. 修改后的总损失

```text
L_total =
    L_ce
  + λ_proto L_proto
  + λ_compact L_compact
  + λ_pair L_pair
  + λ_pair_anchor L_pair_anchor
  + λ_intensity L_intensity
```

推荐初始参数：

```text
proto_loss_weight = 0.5
compact_loss_weight = 0.1

pair_loss_weight = 0.1
pair_margin = 0.3

pair_anchor_loss_weight = 0.05
pair_anchor_upper = 0.20

intensity_loss_weight = 0.05
happy_ce_weight = 1.3
```

---

## 11. 推荐运行参数

```bash
--class_adaptive_ema
--low_conf_classes angry,frustrated
--low_conf_threshold 0.38
--happy_conf_threshold 0.42
--default_ema_conf_threshold 0.45

--use_ema_fallback
--fallback_momentum 0.98

--pair_loss_weight 0.1
--pair_margin 0.3

--pair_anchor_loss_weight 0.05
--pair_anchor_upper 0.20

--happy_ce_weight 1.3

--use_intensity_head
--intensity_loss_weight 0.05
```

如果不稳定，优先降低：

```bash
--pair_loss_weight 0.05
--pair_anchor_loss_weight 0.02
--intensity_loss_weight 0.03
```

---

## 12. 需要新增的日志

### 12.1 混淆类别错误数量

每个 epoch 输出：

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

### 12.2 每类 F1

每个 epoch 输出：

```text
happy_f1
excited_f1
angry_f1
frustrated_f1
neutral_f1
sad_f1
```

### 12.3 EMA 更新统计

继续输出：

```text
ema_update_counts
```

重点观察：

```text
angry 是否仍然为 [0,0,0]
frustrated 是否仍然为 [0,0,0]
```

### 12.4 新增 loss 分项

输出：

```text
loss_pair
loss_pair_anchor
loss_intensity
```

---

## 13. Codex 实现顺序

请 Codex 按以下顺序实现：

```text
1. 修改 EMA 更新：
   - 支持 class-adaptive threshold
   - 支持 fallback soft EMA update

2. 新增 pairwise_confusion_loss：
   - 针对 happy/excited、angry/frustrated、neutral/frustrated、sad/frustrated、sad/neutral

3. 新增 pairwise_anchor_separation_loss：
   - 对混淆类别的锚点中心加弱分离约束

4. 新增 happy 类 CE 权重：
   - happy_ce_weight = 1.3

5. 新增 intensity head：
   - 3 类强度弱标签
   - intensity_loss_weight = 0.05

6. 修改 trainer：
   - 总损失加入 loss_pair、loss_pair_anchor、loss_intensity
   - 日志输出各 loss 分项

7. 修改 evaluation：
   - 输出混淆类别 pair 错误数量
   - 输出每类 F1

8. 保持原 HDSA 主结构不变：
   - 不要删除 OT
   - 不要删除 L_preserve
   - 不要删除高置信 EMA
   - 不要重新改回原 EACL
```

---

## 14. 验收标准

实现后应满足：

```text
1. angry 和 frustrated 的 ema_update_counts 不再长期为 [0,0,0]。
2. happy -> excited 错误数低于 41。
3. angry -> frustrated 与 frustrated -> angry 总数下降。
4. neutral -> frustrated 与 frustrated -> neutral 总数下降。
5. weighted F1 不低于当前 0.6978。
6. 尽量接近或超过当前最高 0.7042。
```

如果出现：

```text
happy F1 上升，但 excited F1 明显下降
```

说明 happy 权重过高，需要降低：

```text
happy_ce_weight: 1.3 → 1.15
```

如果出现整体 F1 下降，优先降低：

```text
pair_loss_weight
pair_anchor_loss_weight
intensity_loss_weight
```

而不是删除整个改进模块。

---

## 15. 论文描述可用文字

```text
在缓解域子锚点塌缩后，模型虽然整体性能明显提升，但在 happy/excited、angry/frustrated、neutral/frustrated 等语义相近类别之间仍存在较明显混淆。针对该问题，本文进一步引入混淆类别定向优化机制。首先，根据验证集混淆矩阵构造易混淆类别对，并在样本级 logits 上加入 pairwise margin loss，使真实类别得分相较于其易混淆类别保持一定间隔。其次，在原型空间中对易混淆类别的锚点中心加入弱分离约束，以增强相近情感类别之间的边界。同时，考虑到 angry 和 frustrated 类别在高置信 EMA 更新中样本不足，本文采用类别自适应 EMA 阈值和 fallback soft update，使难分类类别的域子锚点能够持续适应训练过程。此外，针对 happy 类样本较少且容易被 excited 吞并的问题，适当提高 happy 类交叉熵权重；针对 neutral/frustrated 等强度边界问题，引入轻量情绪强度辅助头，以增强模型对情感强弱差异的感知。通过上述改进，模型能够在保持域子锚点结构稳定的同时，进一步降低相近情感类别之间的误判。
```

---

## 16. 本次改进总结

本次改进围绕“混淆类别定向优化”展开：

```text
1. 类别自适应 EMA 阈值，解决 angry/frustrated 锚点不更新问题。
2. EMA fallback soft update，避免难分类类别锚点长期静止。
3. Pairwise confusion loss，直接惩罚相近类别误判。
4. Pairwise anchor center separation，增强混淆类别原型边界。
5. happy CE 加权，缓解 happy 被 excited 吞并。
6. 情绪强度辅助头，改善 neutral/frustrated 和 happy/excited 强度边界。
7. 新增混淆类别日志，方便后续精确观察。
```

目标：

```text
在保持当前防塌缩效果的基础上，进一步降低 happy/excited、angry/frustrated、neutral/frustrated 等类别之间的混淆，使 weighted F1 稳定超过 0.704。
```
