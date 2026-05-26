# HDSA-ERC 域子锚点塌缩问题改进文档

## 1. 当前问题

当前 HDSA-ERC 已经融合了以下机制：

```text
类别内聚类初始化域子锚点
超球面锚点分离
最优传输 OT 分配
子锚点分类损失 loss_proto
样本—子锚点紧致损失 loss_compact
EMA 动态更新域子锚点
```

但是训练日志显示，同一情感类别下的多个子锚点几乎完全重合：

```text
same_anchor_sim_mean ≈ 1.0
same_anchor_sim_max ≈ 1.0
```

这说明模型虽然形式上有多个域子锚点，但实际上更像：

```text
每个情感类别只有一个中心锚点，被复制成多份
```

这会削弱“域子锚点”的核心作用，因为它不能真正表示同一情感类别内部的不同表达子域。

---

## 2. 塌缩原因

### 2.1 类中心一致性约束过强

当前模型中为了防止同类子锚点跑偏，可能使用了类似约束：

```text
L_center = mean(1 - cos(a_{c,m}, μ_c))
```

其中：

```text
a_{c,m} = 第 c 类第 m 个子锚点
μ_c = 第 c 类所有子锚点的均值中心
```

这个约束会把同一类的多个子锚点全部拉向同一个类中心：

```text
a_{c,1} → μ_c
a_{c,2} → μ_c
a_{c,3} → μ_c
```

如果该约束权重过大，就会直接导致同类子锚点塌缩。

---

### 2.2 OT 分配均衡不等于子域有效

OT / Sinkhorn 会让每个子锚点获得比较均衡的样本质量。  
但是如果同一类子锚点本身已经很相似，那么样本到每个子锚点的距离几乎相同：

```text
cos(z_i, a_{c,1}) ≈ cos(z_i, a_{c,2}) ≈ cos(z_i, a_{c,3})
```

这时 OT 分配会接近：

```text
γ_i ≈ [1/3, 1/3, 1/3]
```

训练日志中每类 assignment counts 几乎完全一样，说明 OT 可能只是被均衡约束强制平均，而不是学到了真实子域。

---

### 2.3 EMA 会放大塌缩

当前 EMA 更新逻辑大致是：

```text
a_{c,m} ← momentum * a_{c,m} + (1 - momentum) * mean(z_i assigned to a_{c,m})
```

如果 OT 分配接近均匀，那么每个子锚点接收到的样本均值也会接近相同：

```text
mean_{c,1} ≈ mean_{c,2} ≈ mean_{c,3}
```

于是 EMA 更新后，各个子锚点会进一步靠近。

---

### 2.4 loss_proto 在均匀分配下会鼓励同类子锚点变相似

如果 OT soft target 接近：

```text
[1/3, 1/3, 1/3]
```

那么 proto loss 会让样本同时靠近本类所有子锚点。  
这样会进一步推动同类子锚点产生相似响应。

---

### 2.5 KMeans 初始化差异没有被保护

域子锚点最初来自 KMeans 聚类中心。  
但是如果训练过程中没有保护这些初始中心，模型为了分类稳定性，可能会把多个子锚点全部推向同一个类中心。

---

## 3. 改进目标

本次改进重点不是继续加强异类锚点分离，而是：

```text
保护同一情感类别内部多个域子锚点之间的差异
```

目标包括：

```text
1. 保留 KMeans 初始化得到的子域差异。
2. 减弱类中心一致性约束。
3. 增强同类子锚点防重合约束。
4. 让 OT 分配具有区分度，而不是对每个样本平均分配。
5. EMA 更新只使用高置信样本，避免均匀分配导致锚点同化。
6. 增加日志，定位塌缩发生阶段。
```

---

## 4. 改进一：新增 KMeans 初始中心保持损失 L_preserve

### 4.1 设计

保留 KMeans 生成的初始锚点：

```text
a^0_{c,m}
```

训练中的当前锚点为：

```text
a_{c,m}
```

新增：

```text
L_preserve = mean_{c,m} [1 - cos(a_{c,m}, a^0_{c,m})]
```

作用：

```text
让每个子锚点不要完全偏离自己的初始聚类中心，
从而保留该子锚点对应的情感表达子域。
```

### 4.2 代码

```python
def anchor_preserve_loss(anchors, init_anchors):
    anchors = F.normalize(anchors, dim=-1)
    init_anchors = F.normalize(init_anchors, dim=-1)
    sim = (anchors * init_anchors).sum(dim=-1)
    return (1.0 - sim).mean()
```

锚点预训练损失改为：

```text
L_anchor =
    L_inter
  + λ_center L_center
  + λ_div L_div
  + λ_rank L_rank
  + λ_preserve L_preserve
```

推荐：

```text
λ_preserve = 0.5 或 1.0
```

---

## 5. 改进二：减弱 L_center，增强 L_div

### 5.1 修改方向

原来如果是：

```text
L_domain = L_center + λ_div L_div
```

建议改成：

```text
L_domain = λ_center L_center + λ_div L_div
```

其中：

```text
λ_center = 0.05 或 0.1
λ_div = 1.0
```

也就是说：

```text
类中心一致性只作为弱约束；
防止同类子锚点重合才是当前重点。
```

### 5.2 L_div 设计

```text
L_div = mean ReLU(cos(a_{c,m}, a_{c,n}) - δ_same), m ≠ n
```

推荐：

```text
δ_same = 0.85 或 0.90
```

### 5.3 代码

```python
def anchor_domain_loss(anchors, same_upper=0.90):
    C, M, D = anchors.shape
    anchors = F.normalize(anchors, dim=-1)

    centers = F.normalize(anchors.mean(dim=1), dim=-1)
    center_sim = torch.einsum("cmd,cd->cm", anchors, centers)
    loss_center = (1.0 - center_sim).mean()

    if M <= 1:
        loss_div = anchors.new_tensor(0.0)
    else:
        flat = anchors.reshape(C * M, D)
        sim = torch.matmul(flat, flat.t())

        labels = torch.arange(C, device=anchors.device).repeat_interleave(M)
        same_mask = labels[:, None] == labels[None, :]

        eye = torch.eye(C * M, dtype=torch.bool, device=anchors.device)
        same_mask = same_mask & (~eye)

        same_sim = sim[same_mask]
        loss_div = F.relu(same_sim - same_upper).mean()

    return loss_center, loss_div
```

---

## 6. 改进三：增加 OT 分配熵和最大概率统计

### 6.1 目的

assignment counts 很平均不一定是好事。  
需要判断每个样本是不是也被平均分到了所有子锚点。

新增统计：

```text
ot_assignment_entropy
ot_assignment_max_prob
```

对于 `M=3`：

```text
如果 max_prob_mean ≈ 0.333，说明几乎完全平均分配。
如果 max_prob_mean > 0.45，说明样本开始明显偏向某个子锚点。
```

### 6.2 代码

```python
def compute_ot_stats(gamma):
    # gamma: [N, M]
    entropy = -(gamma * (gamma + 1e-12).log()).sum(dim=1)
    max_prob = gamma.max(dim=1).values

    return {
        "entropy_mean": entropy.mean().item(),
        "entropy_max": entropy.max().item(),
        "max_prob_mean": max_prob.mean().item(),
        "max_prob_min": max_prob.min().item(),
        "max_prob_max": max_prob.max().item(),
    }
```

日志中每个 epoch 输出：

```text
ot_entropy_mean
ot_max_prob_mean
ot_max_prob_min
ot_max_prob_max
```

---

## 7. 改进四：降低 OT epsilon

### 7.1 目的

Sinkhorn 中：

```text
epsilon 越大，分配越平滑；
epsilon 越小，分配越尖锐。
```

当前如果使用：

```text
ot_epsilon = 0.05
```

建议改为：

```text
ot_epsilon = 0.02
```

如果不稳定，可以折中：

```text
ot_epsilon = 0.03
```

---

## 8. 改进五：对 OT target 做 sharpen

### 8.1 目的

如果 OT soft target 太平均，proto loss 会让样本同时靠近所有同类子锚点。  
因此对 OT target 做 sharpen：

```text
γ_sharp = normalize(γ^q)
```

推荐：

```text
q = 2.0
```

### 8.2 代码

```python
def sharpen_assignment(gamma, power=2.0):
    gamma = gamma.clamp_min(1e-12)
    gamma = gamma ** power
    gamma = gamma / (gamma.sum(dim=1, keepdim=True) + 1e-12)
    return gamma
```

在 proto loss 中使用：

```python
soft_targets_for_loss = sharpen_assignment(
    soft_targets,
    power=args.ot_sharpen_power
)
```

---

## 9. 改进六：EMA 改为高置信 hard update

### 9.1 设计

训练损失可以继续使用 soft assignment。  
但是 EMA 更新不要使用过于平均的 soft target。

EMA 更新规则改为：

```text
1. 对每个样本取 max γ_i。
2. 如果 max γ_i < threshold，该样本不参与 EMA。
3. 如果 max γ_i >= threshold，只更新 argmax γ_i 对应的子锚点。
```

推荐：

```text
ema_conf_threshold = 0.45
prototype_momentum = 0.95
```

对于 `M=3`，平均分配是 0.333。  
阈值 0.45 能过滤掉没有明显子域归属的样本。

### 9.2 代码

```python
@torch.no_grad()
def ema_update_anchors_confident(
    model,
    reps,
    labels,
    soft_targets,
    momentum=0.95,
    threshold=0.45
):
    C, M, D = model.domain_anchors.shape
    reps = F.normalize(reps.detach(), dim=-1)
    target = soft_targets.reshape(-1, C, M)

    update_counts = torch.zeros(C, M, device=reps.device)

    for c in range(C):
        idx = torch.where(labels == c)[0]
        if idx.numel() == 0:
            continue

        z_c = reps[idx]
        gamma_c = target[idx, c]

        max_prob, hard_m = gamma_c.max(dim=1)
        keep = max_prob >= threshold

        if keep.sum().item() == 0:
            continue

        z_keep = z_c[keep]
        m_keep = hard_m[keep]

        for m in range(M):
            m_idx = torch.where(m_keep == m)[0]
            if m_idx.numel() == 0:
                continue

            mean = z_keep[m_idx].mean(dim=0)
            mean = F.normalize(mean, dim=-1)

            old = model.domain_anchors[c, m]
            new = momentum * old + (1.0 - momentum) * mean
            model.domain_anchors[c, m] = F.normalize(new, dim=-1)

            update_counts[c, m] += m_idx.numel()

    return update_counts
```

---

## 10. 改进七：定位塌缩发生阶段

必须在三个阶段输出 anchor stats。

### 10.1 KMeans 后

在 `generate_domain_anchors.py` 保存前输出：

```text
kmeans_same_anchor_sim_mean
kmeans_same_anchor_sim_max
kmeans_diff_anchor_sim_mean
kmeans_diff_anchor_sim_max
```

### 10.2 锚点预训练后

在 `pretrain_hyp_domain_anchors.py` 结束时输出：

```text
pretrain_same_anchor_sim_mean
pretrain_same_anchor_sim_max
pretrain_diff_anchor_sim_mean
pretrain_diff_anchor_sim_max
```

### 10.3 主训练加载后

模型初始化后输出：

```text
loaded_same_anchor_sim_mean
loaded_same_anchor_sim_max
loaded_diff_anchor_sim_mean
loaded_diff_anchor_sim_max
```

判断逻辑：

```text
如果 KMeans 后 same sim 接近 1：
    聚类没产生有效子域，或聚类代码/维度有问题。

如果 KMeans 正常，预训练后接近 1：
    锚点预训练阶段导致塌缩。

如果预训练后正常，加载后接近 1：
    保存或加载代码把锚点复制错了。

如果加载后正常，epoch 1 后接近 1：
    主训练中的 OT + EMA 导致塌缩。
```

---

## 11. 新增参数

在 `run.py` 和相关脚本中新增：

```python
parser.add_argument("--preserve_weight", type=float, default=0.5)
parser.add_argument("--center_weight", type=float, default=0.1)
parser.add_argument("--div_weight", type=float, default=1.0)
parser.add_argument("--same_upper", type=float, default=0.90)

parser.add_argument("--ot_sharpen_power", type=float, default=2.0)
parser.add_argument("--ema_conf_threshold", type=float, default=0.45)
```

建议修改默认值：

```python
parser.add_argument("--ot_epsilon", type=float, default=0.02)
parser.add_argument("--prototype_momentum", type=float, default=0.95)
```

---

## 12. 修改后的训练流程

主训练流程改成：

```text
1. 前向得到 z、logits、anchors。
2. 按类别执行 OT 分配，得到 soft_targets。
3. 记录 OT entropy 和 max_prob。
4. 对 soft_targets 做 sharpen，用于 loss_proto。
5. 计算 loss_ce、loss_proto、loss_compact。
6. 反向传播更新 encoder、map_function、classifier。
7. 使用高置信 OT hard assignment 执行 EMA anchor update。
8. 输出 anchor stats、OT stats、EMA update counts。
```

伪代码：

```python
soft_targets, assigned_anchor, assignment_counts, ot_stats = ot_assign_by_class(...)

soft_targets_for_loss = sharpen_assignment(
    soft_targets,
    power=args.ot_sharpen_power
)

loss_proto = soft_cross_entropy(proto_logits, soft_targets_for_loss)

loss_compact = ((1.0 - cos(z, assigned_anchor.detach())) ** 2).mean()

loss = (
    loss_ce
    + args.proto_loss_weight * loss_proto
    + args.compact_loss_weight * loss_compact
)

loss.backward()
optimizer.step()

ema_counts = ema_update_anchors_confident(
    model=model,
    reps=z.detach(),
    labels=labels,
    soft_targets=soft_targets.detach(),
    momentum=args.prototype_momentum,
    threshold=args.ema_conf_threshold
)
```

---

## 13. 推荐实验参数

下一次实验建议使用：

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

如果同类子锚点仍塌缩：

```bash
--center_weight 0.05
--div_weight 1.5
--preserve_weight 1.0
--same_upper 0.85
```

如果 OT 太硬导致性能下降：

```bash
--ot_epsilon 0.03
--ot_sharpen_power 1.5
--ema_conf_threshold 0.40
```

---

## 14. 关键观察指标

### 14.1 子锚点是否仍塌缩

重点看：

```text
same_anchor_sim_mean
same_anchor_sim_max
```

理想情况：

```text
same_anchor_sim_mean 不再接近 1.0
same_anchor_sim_mean 能降到 0.85~0.95 更合理
```

### 14.2 OT 是否有区分度

重点看：

```text
ot_max_prob_mean
ot_entropy_mean
```

对于 `M=3`：

```text
ot_max_prob_mean ≈ 0.333：完全平均
ot_max_prob_mean > 0.45：有一定区分度
```

### 14.3 EMA 是否真的更新了每个子锚点

重点看：

```text
ema_update_counts
```

如果大量为 0，说明阈值过高或锚点吸引不到样本。

---

## 15. Codex 实现顺序

请 Codex 按下面顺序改：

```text
1. 在 generate_domain_anchors.py 中保存 init_anchors，并输出 KMeans 后 anchor stats。
2. 在 pretrain_hyp_domain_anchors.py 中加入 L_preserve。
3. 将 L_domain 拆成 L_center 和 L_div，并降低 center 权重。
4. 增加 anchor similarity 统计函数，用于 KMeans 后、预训练后、加载后、每个 epoch。
5. 在 sinkhorn / ot_assign_by_class 中增加 OT entropy 和 max_prob 统计。
6. 增加 sharpen_assignment 函数，用于 loss_proto。
7. 修改 EMA 更新为 high-confidence hard EMA update。
8. 在 run.py 中加入新参数。
9. 在 trainer.py 日志中输出：
   - loss_ce
   - loss_proto
   - loss_compact
   - same/diff anchor stats
   - ot_entropy / ot_max_prob
   - ema_update_counts
10. 确保 use_hdsa=False 时旧模型逻辑不受影响。
```

---

## 16. 论文描述可用文字

```text
在初步实验中，我们发现直接结合最优传输分配与域子锚点更新容易出现子锚点塌缩现象，即同一情感类别下的多个子锚点逐渐退化为高度相似的单一类别中心。为缓解该问题，本文进一步设计了域子锚点差异保持机制。首先，在类别内聚类得到初始域子锚点后，模型保留每个子锚点的初始聚类中心，并引入聚类中心保持损失，约束训练后的子锚点不偏离其初始语义子域。其次，在类内域约束中降低类中心一致性项的权重，并增强同类子锚点防重合约束，使同一情感类别下的多个子锚点既保持类别一致性，又具有一定表达差异。最后，在最优传输分配阶段记录分配熵和最大分配概率，并采用低熵化分配与高置信 EMA 更新策略，使子锚点主要由与其匹配度较高的样本更新，从而避免均匀软分配导致的子锚点同化。通过上述改进，模型能够更有效地保持情感类别内部的多子域结构。
```

---

## 17. 本次改进总结

本次改进主要解决：

```text
域子锚点塌缩
OT 平均分配
EMA 同化更新
KMeans 子域差异丢失
```

具体改动：

```text
1. 新增 L_preserve，保持 KMeans 初始子域。
2. 降低 L_center 权重，避免同类锚点被拉到同一个中心。
3. 增强 L_div，防止同类子锚点过度相似。
4. 降低 OT epsilon，让分配更有区分度。
5. 对 OT target 做 sharpen，避免 loss_proto 过于平均。
6. EMA 更新改为高置信 hard assignment 更新。
7. 增加 OT entropy、max_prob、EMA counts 等日志，定位塌缩发生阶段。
```

最终目标：

```text
让每个情感类别下的多个域子锚点真正对应不同表达子域，而不是退化为同一个情感中心的重复副本。
```
