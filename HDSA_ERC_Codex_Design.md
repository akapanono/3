# HDSA-ERC 具体设计文稿：结合域子锚点与 HMPEAE 的对话情感识别模型

> 文件用途：给 Codex 作为完整实现说明。  
> 模型目标：在当前 ERC 项目基础上，保留“域子锚点”思想，同时吸收 HMPEAE 的“超球面多原型 + 最优传输分配 + 紧致学习 + EMA 更新”机制，形成一个结构清晰、可直接落地的模型。

---

## 1. 模型名称

中文名称：

```text
基于超球面域子锚点与最优传输的对话情感识别模型
```

英文名称：

```text
Hyperspherical Domain Sub-anchor with Optimal Transport for ERC
```

简称：

```text
HDSA-ERC
```

其中：

```text
H = Hyperspherical，超球面表示空间
D = Domain，情感类别内部的表达子域
S = Sub-anchor，每个情感类别下的多个子锚点
A = Anchor，情感锚点 / 原型
OT = Optimal Transport，最优传输分配
ERC = Emotion Recognition in Conversation
```

---

## 2. 模型核心思想

本模型只保留一条主线：

```text
预编码目标话语
→ 类别内聚类得到域子锚点初始值
→ 参考 HMPEAE 进行超球面锚点优化
→ ERC 主模型编码话语表示
→ 用最优传输分配样本到本类域子锚点
→ 用情感分类损失、子锚点分类损失、紧致损失联合训练
→ 根据 OT 分配结果用 EMA 更新域子锚点
→ 推理时结合分类头与锚点相似度得到最终情感类别
```

模型要解决两个问题：

```text
1. 同一情感类别内部表达差异大。
   例如 angry 可能包括指责型、抱怨型、压抑型等不同表达子域。

2. 不同情感类别之间边界模糊。
   例如 angry/frustrated、happy/excited、sad/neutral 容易混淆。
```

因此，每个情感类别不再只对应一个锚点，而是对应多个由类别内聚类得到的域子锚点。随后参考 HMPEAE，将这些域子锚点放到超球面空间中进行类间分离，并用最优传输控制样本到子锚点的分配。

---

## 3. 与 HMPEAE 的对应关系

| HMPEAE 原论文概念 | 本模型 HDSA-ERC 对应概念 |
|---|---|
| Event role | Emotion label |
| Argument representation | Utterance emotion representation |
| Role prototype | Emotion domain sub-anchor |
| Multiple prototypes per role | Multiple domain sub-anchors per emotion |
| Hyperspherical prototype separation | Hyperspherical emotion anchor separation |
| Argument-prototype assignment | Utterance-sub-anchor assignment |
| Optimal Transport assignment | OT-based sample-anchor assignment |
| Compactness loss | Sample-sub-anchor compactness loss |
| EMA prototype update | EMA domain sub-anchor update |
| TabEAE backbone | Sup-SimCSE-RoBERTa / RoBERTa ERC encoder |

---

## 4. 整体流程

完整流程如下：

```text
Step 1. 构造对话上下文 prompt。
Step 2. 使用预训练语言模型编码目标话语，取 <mask> 位置表示。
Step 3. 使用映射层将表示映射到低维情感锚点空间。
Step 4. 在训练集上提取所有样本的映射表示。
Step 5. 按情感类别进行类别内聚类，得到每个情感类别的多个域子锚点初始值。
Step 6. 使用超球面锚点优化目标，进一步调整域子锚点分布。
Step 7. 训练 ERC 主模型时，对每个类别内部的样本和域子锚点做 OT 分配。
Step 8. 根据 OT 分配结果计算子锚点分类损失和紧致损失。
Step 9. 使用情感分类损失、子锚点分类损失、紧致损失联合优化模型。
Step 10. 根据 OT 分配结果，用 EMA 动态更新域子锚点。
Step 11. 推理时计算分类头 logits 和锚点相似度 logits，融合得到最终预测。
```

---

## 5. 输入构造

### 5.1 对话上下文格式

对于一段对话，将历史话语整理成：

```text
{speaker} says: {utterance_text}
```

例如：

```text
A says: I cannot believe you did that.
B says: I already apologized.
A says: That is not enough.
```

### 5.2 目标话语 Prompt

目标话语使用当前项目已有形式：

```text
For utterance: {target_text} {speaker} feels <mask>
```

最终输入：

```text
A says: I cannot believe you did that.
B says: I already apologized.
A says: That is not enough.
For utterance: I already apologized. B feels <mask>
```

模型取 `<mask>` 位置的 hidden state 作为目标话语表示。

---

## 6. 预编码器与映射层

### 6.1 Encoder

使用：

```text
Sup-SimCSE-RoBERTa-large
```

或：

```text
RoBERTa-large
```

输入为 prompt 后，得到：

```text
H = Encoder(X)
```

取 `<mask>` 位置表示：

```text
h_i = H[mask_pos]
```

其中：

```text
h_i ∈ R^hidden_dim
```

如果使用 RoBERTa-large：

```text
hidden_dim = 1024
```

### 6.2 Mapping MLP

将 PLM 表示映射到情感锚点空间：

```text
z_i = f_map(h_i)
```

建议结构：

```python
self.map_function = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, anchor_dim)
)
```

推荐：

```text
anchor_dim = 256 或 512
dropout = 0.1
```

映射后归一化：

```python
z_i = F.normalize(z_i, dim=-1)
```

---

## 7. 域子锚点初始化：类别内聚类

### 7.1 为什么要聚类

HMPEAE 中多原型用于表达同一类别内部的语义差异。本模型将这个思想迁移到 ERC，但进一步引入“域子锚点”：

```text
每个情感类别内部可能存在多个表达子域。
这些子域不靠人工定义，而是由训练集样本表示聚类得到。
```

### 7.2 聚类输入

先使用预编码器和映射层，提取训练集所有样本的表示：

```text
z_i = normalize(f_map(Encoder(x_i)[mask_pos]))
```

得到：

```text
Z = {z_1, z_2, ..., z_N}
```

按真实情感标签分组：

```text
Z_c = {z_i | y_i = c}
```

### 7.3 类别内聚类

对每个类别 `c` 内部做 KMeans：

```text
KMeans(Z_c, M) → {a_{c,1}, a_{c,2}, ..., a_{c,M}}
```

其中：

```text
C = 情感类别数
M = 每类域子锚点数
D = anchor_dim
```

最终得到：

```text
A ∈ R^{C × M × D}
```

推荐配置：

```text
IEMOCAP:
C = 6
M = 3
D = 256 或 512
```

### 7.4 聚类脚本输出

新增脚本：

```text
src/anchors/generate_domain_anchors.py
```

输出文件：

```text
domain_anchors/{dataset_name}_M{num_subanchors}.pt
```

保存格式：

```python
{
    "anchors": anchors,              # Tensor [C, M, D]
    "counts": counts,                # Tensor [C, M]
    "label2id": label2id,
    "id2label": id2label,
    "anchor_dim": anchor_dim,
    "num_subanchors": M,
    "source": "class_wise_kmeans"
}
```

### 7.5 代码伪实现

```python
def extract_representations(model, dataloader, device):
    model.eval()
    all_reps = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            out = model.encode(batch)
            z = F.normalize(out["z"], dim=-1)
            all_reps.append(z.cpu())
            all_labels.append(batch["labels"].cpu())

    return torch.cat(all_reps, dim=0), torch.cat(all_labels, dim=0)


def build_domain_anchors(reps, labels, num_classes, num_subanchors):
    anchors = []
    counts = []

    for c in range(num_classes):
        cls_reps = reps[labels == c]
        assert len(cls_reps) > 0, f"class {c} has no samples"

        k = min(num_subanchors, len(cls_reps))

        km = KMeans(n_clusters=k, random_state=0, n_init="auto")
        assign = km.fit_predict(cls_reps.numpy())
        centers = torch.tensor(km.cluster_centers_, dtype=torch.float)
        centers = F.normalize(centers, dim=-1)

        if k < num_subanchors:
            pad = centers[:1].repeat(num_subanchors - k, 1)
            centers = torch.cat([centers, pad], dim=0)

        cls_counts = torch.zeros(num_subanchors, dtype=torch.long)
        for idx in range(k):
            cls_counts[idx] = int((assign == idx).sum())

        anchors.append(centers)
        counts.append(cls_counts)

    anchors = torch.stack(anchors, dim=0)
    counts = torch.stack(counts, dim=0)
    return anchors, counts
```

---

## 8. 超球面域子锚点优化

聚类得到的域子锚点只是初始值，可能仍然存在异类锚点过近的问题。因此，需要参考 HMPEAE 对锚点进行超球面优化。

所有锚点先归一化：

```python
A = F.normalize(A, dim=-1)
```

超球面锚点优化包括三部分：

```text
L_anchor = L_inter + λ_domain L_domain + λ_rank L_rank
```

---

## 9. 类间锚点分离损失 L_inter

### 9.1 目标

不同情感类别的域子锚点应该分开。对于每个子锚点，找到最相似的异类子锚点，并降低它们的余弦相似度。

### 9.2 公式

对每个子锚点 `a_{c,m}`：

```text
hardest_neg(a_{c,m}) = max_{c'≠c,n} cos(a_{c,m}, a_{c',n})
```

损失：

```text
L_inter = mean_{c,m} hardest_neg(a_{c,m})
```

训练时最小化它。

### 9.3 代码

```python
def anchor_inter_loss(anchors):
    C, M, D = anchors.shape
    anchors = F.normalize(anchors, dim=-1)

    flat = anchors.reshape(C * M, D)
    sim = torch.matmul(flat, flat.t())

    labels = torch.arange(C, device=anchors.device).repeat_interleave(M)
    diff_mask = labels[:, None] != labels[None, :]

    hardest = sim.masked_fill(~diff_mask, -1e4).max(dim=1).values
    return hardest.mean()
```

---

## 10. 域子锚点类内约束 L_domain

### 10.1 设计原则

这里不能简单照搬 HMPEAE 的“同类原型尽量靠近”。因为本模型的子锚点代表同一情感类别内部的不同表达子域，不能全部重合。

因此，类内约束采用：

```text
类内一致性 + 域内差异性
```

也就是：

```text
同一情感类别的多个子锚点不能离本类太远；
但也不能全部塌缩到一起。
```

### 10.2 类中心一致性 L_center

先计算每个情感类别的锚点中心：

```text
μ_c = mean_m a_{c,m}
```

让同一类子锚点围绕本类中心：

```text
L_center = mean_{c,m} [1 - cos(a_{c,m}, μ_c)]
```

作用：

```text
保证同一情感类别的子锚点仍然属于同一情感区域。
```

### 10.3 防塌缩差异约束 L_div

同一类内部的子锚点不能过于相似。如果相似度超过阈值，则惩罚：

```text
L_div = mean ReLU(cos(a_{c,m}, a_{c,n}) - δ_same), m ≠ n
```

推荐：

```text
δ_same = 0.90
```

注意这里不是强行把同类子锚点拉得很远，只是防止完全重合。

### 10.4 L_domain

```text
L_domain = L_center + λ_div L_div
```

推荐：

```text
λ_div = 0.5
```

### 10.5 代码

```python
def anchor_domain_loss(anchors, same_upper=0.90, div_weight=0.5):
    C, M, D = anchors.shape
    anchors = F.normalize(anchors, dim=-1)

    centers = F.normalize(anchors.mean(dim=1), dim=-1)  # [C, D]
    center_sim = torch.einsum("cmd,cd->cm", anchors, centers)
    loss_center = (1.0 - center_sim).mean()

    if M <= 1:
        return loss_center

    flat = anchors.reshape(C * M, D)
    sim = torch.matmul(flat, flat.t())

    labels = torch.arange(C, device=anchors.device).repeat_interleave(M)
    same_mask = labels[:, None] == labels[None, :]

    eye = torch.eye(C * M, dtype=torch.bool, device=anchors.device)
    same_mask = same_mask & (~eye)

    same_sim = sim[same_mask]
    loss_div = F.relu(same_sim - same_upper).mean()

    return loss_center + div_weight * loss_div
```

---

## 11. 情感标签语义排序损失 L_rank

### 11.1 目标

HMPEAE 使用标签语义先验，使原型距离符合类别语义关系。本模型将 role label 换成 emotion label。

例如：

```text
angry 与 frustrated 语义较近；
happy 与 excited 语义较近；
sad 与 excited 语义较远。
```

锚点空间也应该大致保持这种关系。

### 11.2 情感标签描述

为每个情感类别构造描述句：

```python
emotion_descriptions = {
    "angry": "The speaker feels angry and irritated.",
    "frustrated": "The speaker feels frustrated and annoyed.",
    "happy": "The speaker feels happy and pleased.",
    "excited": "The speaker feels excited and energetic.",
    "neutral": "The speaker feels neutral and calm.",
    "sad": "The speaker feels sad and disappointed."
}
```

用同一个 encoder 或 sentence encoder 得到标签语义表示：

```text
e_c = Encoder(description_c)
```

然后归一化。

### 11.3 排序约束

每个类别的锚点中心：

```text
μ_c = mean_m a_{c,m}
```

标签语义距离：

```text
d_label(i,j) = 1 - cos(e_i, e_j)
```

锚点类别中心距离：

```text
d_anchor(i,j) = 1 - cos(μ_i, μ_j)
```

对于三元组 `(i,j,k)`：

```text
如果 d_label(i,j) <= d_label(i,k)，
则希望 d_anchor(i,j) <= d_anchor(i,k)。
```

使用 RankNet 风格损失。

### 11.4 代码简化实现

```python
def anchor_rank_loss(anchors, label_embeddings):
    C, M, D = anchors.shape

    centers = F.normalize(anchors.mean(dim=1), dim=-1)
    label_embeddings = F.normalize(label_embeddings, dim=-1)

    anchor_dist = 1.0 - torch.matmul(centers, centers.t())
    label_dist = 1.0 - torch.matmul(label_embeddings, label_embeddings.t())

    losses = []
    for i in range(C):
        for j in range(C):
            for k in range(C):
                if i == j or i == k or j == k:
                    continue

                target = (label_dist[i, j] <= label_dist[i, k]).float()
                score = torch.sigmoid(anchor_dist[i, j] - anchor_dist[i, k])
                loss = F.binary_cross_entropy(score, target)
                losses.append(loss)

    if not losses:
        return anchors.new_tensor(0.0)

    return torch.stack(losses).mean()
```

---

## 12. 锚点预训练脚本

新增脚本：

```text
src/anchors/pretrain_hyp_domain_anchors.py
```

输入：

```text
cluster initialized domain anchors
emotion label embeddings
```

输出：

```text
hyperspherical optimized domain anchors
```

训练目标：

```text
L_anchor = L_inter + λ_domain L_domain + λ_rank L_rank
```

推荐参数：

```text
anchor_pretrain_epochs = 1000
anchor_pretrain_lr = 0.1
optimizer = SGD
momentum = 0.9
λ_domain = 1.0
λ_rank = 1.0
same_upper = 0.90
```

伪代码：

```python
anchors = nn.Parameter(init_anchors.clone())
optimizer = torch.optim.SGD([anchors], lr=args.lr, momentum=0.9)

for step in range(args.epochs):
    norm_anchors = F.normalize(anchors, dim=-1)

    loss_inter = anchor_inter_loss(norm_anchors)
    loss_domain = anchor_domain_loss(norm_anchors)
    loss_rank = anchor_rank_loss(norm_anchors, label_embeddings)

    loss = loss_inter + args.domain_weight * loss_domain + args.rank_weight * loss_rank

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        anchors.copy_(F.normalize(anchors, dim=-1))
```

保存：

```python
{
    "anchors": F.normalize(anchors.detach().cpu(), dim=-1),
    "label_embeddings": label_embeddings.cpu(),
    "args": vars(args),
    "source": "kmeans_init_plus_hyperspherical_pretrain"
}
```

---

## 13. ERC 主训练：最优传输分配

### 13.1 为什么使用 OT

如果直接把样本分配给最近的子锚点，容易出现：

```text
1. 所有样本都挤到一个子锚点；
2. 其他子锚点几乎不用；
3. 域子锚点失去意义。
```

HDSA-ERC 采用 OT 分配，对每个情感类别内部的样本和该类子锚点进行分配。

### 13.2 按类别构造代价矩阵

对于一个 batch 中真实标签为 `c` 的样本：

```text
Z_c = {z_i | y_i = c}
```

该类域子锚点：

```text
A_c = {a_{c,1}, ..., a_{c,M}}
```

构造 cost：

```text
Cost(i,m) = 1 - cos(z_i, a_{c,m})
```

### 13.3 Sinkhorn 代码

新增文件：

```text
src/anchors/sinkhorn.py
```

代码：

```python
def sinkhorn(cost, epsilon=0.05, n_iters=50):
    """
    cost: [N, M]
    return gamma: [N, M], row-normalized soft assignment
    """
    N, M = cost.shape
    device = cost.device

    a = torch.full((N,), 1.0 / N, device=device)
    b = torch.full((M,), 1.0 / M, device=device)

    K = torch.exp(-cost / epsilon).clamp_min(1e-12)

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(n_iters):
        u = a / (K @ v + 1e-12)
        v = b / (K.t() @ u + 1e-12)

    gamma = torch.diag(u) @ K @ torch.diag(v)
    gamma = gamma / (gamma.sum(dim=1, keepdim=True) + 1e-12)
    return gamma
```

### 13.4 Batch 内分配函数

```python
def ot_assign_by_class(reps, labels, anchors, epsilon=0.05, n_iters=50):
    B, D = reps.shape
    C, M, _ = anchors.shape
    device = reps.device

    reps = F.normalize(reps, dim=-1)
    anchors = F.normalize(anchors, dim=-1)

    soft_targets = torch.zeros(B, C * M, device=device)
    assigned_anchor = torch.zeros(B, D, device=device)
    assignment_counts = torch.zeros(C, M, device=device)

    for c in range(C):
        idx = torch.where(labels == c)[0]
        if idx.numel() == 0:
            continue

        z_c = reps[idx]
        a_c = anchors[c]

        cost = 1.0 - torch.matmul(z_c, a_c.t())
        gamma = sinkhorn(cost, epsilon=epsilon, n_iters=n_iters)

        start = c * M
        soft_targets[idx, start:start + M] = gamma
        assigned_anchor[idx] = gamma @ a_c
        assignment_counts[c] += gamma.sum(dim=0)

    assigned_anchor = F.normalize(assigned_anchor, dim=-1)
    return soft_targets, assigned_anchor, assignment_counts
```

---

## 14. ERC 主模型结构

### 14.1 模型输出

主模型输入 batch 后输出：

```python
{
    "logits_cls": logits_cls,        # [B, C]
    "logits_anchor": logits_anchor,  # [B, C]
    "logits": logits,                # [B, C]
    "z": z,                          # [B, D]
    "anchors": anchors               # [C, M, D]
}
```

### 14.2 Anchor logits

计算样本与所有子锚点相似度：

```python
scores = torch.einsum("bd,cmd->bcm", z, anchors)
```

聚合为类别分数：

```python
logits_anchor = torch.logsumexp(scores / tau, dim=-1) * tau
```

### 14.3 最终 logits

融合分类头和锚点头：

```text
logits = (1 - β) logits_cls + β logits_anchor
```

推荐：

```text
β = 0.3
```

原因：

```text
分类头提供稳定预测；
锚点头提供结构化原型边界；
不要完全依赖锚点头，避免锚点空间波动直接影响预测。
```

### 14.4 模型伪代码

```python
class HDSAERCModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.bert_path)

        self.map_function = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LayerNorm(args.hidden_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.hidden_dim, args.anchor_dim)
        )

        self.classifier = nn.Linear(args.anchor_dim, args.num_classes)

        obj = torch.load(args.domain_anchor_path, map_location="cpu")
        anchors = obj["anchors"] if isinstance(obj, dict) else obj

        self.register_buffer("domain_anchors", F.normalize(anchors.float(), dim=-1))

        self.temperature = args.anchor_temperature
        self.anchor_logit_weight = args.anchor_logit_weight

    def get_anchors(self):
        return F.normalize(self.domain_anchors, dim=-1)

    def forward(self, input_ids, attention_mask, mask_pos):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        h = hidden[batch_idx, mask_pos]

        z = self.map_function(h)
        z = F.normalize(z, dim=-1)

        logits_cls = self.classifier(z)

        anchors = self.get_anchors()
        scores = torch.einsum("bd,cmd->bcm", z, anchors)
        logits_anchor = torch.logsumexp(scores / self.temperature, dim=-1) * self.temperature

        logits = (
            (1.0 - self.anchor_logit_weight) * logits_cls
            + self.anchor_logit_weight * logits_anchor
        )

        return {
            "logits": logits,
            "logits_cls": logits_cls,
            "logits_anchor": logits_anchor,
            "z": z,
            "anchors": anchors,
            "scores": scores
        }
```

---

## 15. 主训练损失

### 15.1 情感分类损失 L_ce

```python
loss_ce = F.cross_entropy(logits, labels)
```

### 15.2 子锚点分类损失 L_proto

通过 OT 得到 soft target：

```text
soft_targets ∈ R^{B × (C×M)}
```

计算样本与所有子锚点的相似度：

```python
flat_anchors = anchors.reshape(C * M, D)
proto_logits = torch.matmul(z, flat_anchors.t()) / proto_temperature
```

使用 soft cross entropy：

```python
def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()
```

```python
loss_proto = soft_cross_entropy(proto_logits, soft_targets)
```

作用：

```text
让样本不仅预测正确情感类别，
还要靠近该类别下 OT 分配出的具体域子锚点。
```

### 15.3 紧致损失 L_compact

OT 分配得到 assigned anchor：

```text
a_i^+ = Σ_m γ_i,m a_{y_i,m}
```

损失：

```text
L_compact = mean_i (1 - cos(z_i, a_i^+))^2
```

代码：

```python
cos = F.cosine_similarity(z, assigned_anchor, dim=-1)
loss_compact = ((1.0 - cos) ** 2).mean()
```

### 15.4 总损失

主训练总损失：

```text
L_total = L_ce + λ_proto L_proto + λ_compact L_compact
```

推荐：

```text
λ_proto = 0.5
λ_compact = 0.1
```

注意：

```text
这里不再把 SupCon 作为主损失。
因为该模型的主线是 HMPEAE-style prototype assignment，而不是 EACL-style SupCon。
```

---

## 16. EMA 更新域子锚点

### 16.1 更新原则

域子锚点不作为普通可训练参数直接进入 optimizer。它们在训练过程中通过 OT 分配结果进行 EMA 更新。

### 16.2 更新方式

对每个类别 `c` 和子锚点 `m`：

```text
a_{c,m} ← α a_{c,m} + (1 - α) mean(z_i assigned to a_{c,m})
```

由于 OT 是软分配，使用加权均值：

```text
mean_{c,m} = Σ_i γ_i,m z_i / Σ_i γ_i,m
```

然后归一化：

```text
a_{c,m} ← normalize(a_{c,m})
```

推荐：

```text
α = 0.9
```

### 16.3 代码

```python
@torch.no_grad()
def ema_update_anchors(model, reps, labels, soft_targets, momentum=0.9):
    C, M, D = model.domain_anchors.shape
    reps = F.normalize(reps.detach(), dim=-1)

    target = soft_targets.reshape(-1, C, M)

    for c in range(C):
        idx = torch.where(labels == c)[0]
        if idx.numel() == 0:
            continue

        z_c = reps[idx]
        gamma_c = target[idx, c]

        for m in range(M):
            weight = gamma_c[:, m]
            mass = weight.sum()

            if mass.item() <= 1e-6:
                continue

            mean = (weight[:, None] * z_c).sum(dim=0) / mass
            mean = F.normalize(mean, dim=-1)

            old = model.domain_anchors[c, m]
            new = momentum * old + (1.0 - momentum) * mean
            model.domain_anchors[c, m] = F.normalize(new, dim=-1)
```

---

## 17. 主训练伪代码

```python
for batch in train_loader:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        mask_pos=batch["mask_pos"]
    )

    logits = outputs["logits"]
    z = outputs["z"]
    anchors = outputs["anchors"]
    labels = batch["labels"]

    loss_ce = F.cross_entropy(logits, labels)

    soft_targets, assigned_anchor, assignment_counts = ot_assign_by_class(
        reps=z.detach(),
        labels=labels,
        anchors=anchors.detach(),
        epsilon=args.ot_epsilon,
        n_iters=args.ot_iters
    )

    flat_anchors = anchors.reshape(args.num_classes * args.num_subanchors, args.anchor_dim)
    proto_logits = torch.matmul(z, flat_anchors.t()) / args.proto_temperature

    loss_proto = soft_cross_entropy(proto_logits, soft_targets)

    cos = F.cosine_similarity(z, assigned_anchor.detach(), dim=-1)
    loss_compact = ((1.0 - cos) ** 2).mean()

    loss = (
        loss_ce
        + args.proto_loss_weight * loss_proto
        + args.compact_loss_weight * loss_compact
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    ema_update_anchors(
        model=model,
        reps=z.detach(),
        labels=labels,
        soft_targets=soft_targets.detach(),
        momentum=args.prototype_momentum
    )
```

---

## 18. 推理流程

推理时不做 OT 分配，只做编码与预测。

```text
输入对话 prompt
→ Encoder
→ <mask> hidden state
→ map_function 得到 z
→ 分类头得到 logits_cls
→ z 与所有域子锚点计算相似度
→ LogSumExp 聚合得到 logits_anchor
→ logits = (1-β) logits_cls + β logits_anchor
→ argmax 得到最终情感类别
```

---

## 19. 需要新增或修改的文件

### 19.1 新增文件

```text
src/anchors/generate_domain_anchors.py
src/anchors/pretrain_hyp_domain_anchors.py
src/anchors/sinkhorn.py
src/model/hdsa_erc_model.py
src/model/hdsa_losses.py
```

### 19.2 修改文件

```text
src/run.py
src/trainer/trainer.py
src/model/model.py
run.sh
```

如果不想新增 `hdsa_erc_model.py`，也可以在当前 `CLModel` 中增加 HDSA 分支。但是建议单独新增模型类，避免和旧 EACL 逻辑混在一起。

---

## 20. 新增参数

在 `src/run.py` 中新增：

```python
parser.add_argument("--use_hdsa", action="store_true")
parser.add_argument("--domain_anchor_path", type=str, default=None)

parser.add_argument("--num_subanchors", type=int, default=3)
parser.add_argument("--anchor_dim", type=int, default=256)

parser.add_argument("--anchor_temperature", type=float, default=0.1)
parser.add_argument("--proto_temperature", type=float, default=0.1)
parser.add_argument("--anchor_logit_weight", type=float, default=0.3)

parser.add_argument("--proto_loss_weight", type=float, default=0.5)
parser.add_argument("--compact_loss_weight", type=float, default=0.1)

parser.add_argument("--ot_epsilon", type=float, default=0.05)
parser.add_argument("--ot_iters", type=int, default=50)

parser.add_argument("--prototype_momentum", type=float, default=0.9)
```

锚点预训练脚本新增：

```python
parser.add_argument("--init_anchor_path", type=str, required=True)
parser.add_argument("--output_anchor_path", type=str, required=True)
parser.add_argument("--anchor_pretrain_epochs", type=int, default=1000)
parser.add_argument("--anchor_pretrain_lr", type=float, default=0.1)
parser.add_argument("--domain_weight", type=float, default=1.0)
parser.add_argument("--rank_weight", type=float, default=1.0)
parser.add_argument("--same_upper", type=float, default=0.90)
```

---

## 21. 推荐超参数

以 IEMOCAP 为例：

```text
num_subanchors = 3
anchor_dim = 256
anchor_temperature = 0.1
proto_temperature = 0.1
anchor_logit_weight = 0.3

proto_loss_weight = 0.5
compact_loss_weight = 0.1

ot_epsilon = 0.05
ot_iters = 50
prototype_momentum = 0.9

plm_lr = 1e-5
other_lr = 4e-4
batch_size = 8
epochs = 8
max_grad_norm = 5.0
```

锚点预训练：

```text
anchor_pretrain_epochs = 1000
anchor_pretrain_lr = 0.1
domain_weight = 1.0
rank_weight = 1.0
same_upper = 0.90
```

---

## 22. 运行流程

### 22.1 训练基础编码模型并提取表示

可以先使用当前模型训练一个基础 encoder。然后用该 checkpoint 提取 `z_i` 并生成 KMeans 域子锚点。

```bash
python src/anchors/generate_domain_anchors.py \
  --dataset_name IEMOCAP \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --checkpoint_path ./saved_models/IEMOCAP/model_.pkl \
  --num_subanchors 3 \
  --anchor_dim 256 \
  --output_path ./domain_anchors/IEMOCAP_M3_kmeans.pt
```

### 22.2 预训练超球面域子锚点

```bash
python src/anchors/pretrain_hyp_domain_anchors.py \
  --init_anchor_path ./domain_anchors/IEMOCAP_M3_kmeans.pt \
  --output_anchor_path ./domain_anchors/IEMOCAP_M3_hyp.pt \
  --anchor_pretrain_epochs 1000 \
  --anchor_pretrain_lr 0.1 \
  --domain_weight 1.0 \
  --rank_weight 1.0 \
  --same_upper 0.90
```

### 22.3 训练 HDSA-ERC 主模型

```bash
CUDA_VISIBLE_DEVICES=3 python src/run.py \
  --use_hdsa \
  --dataset_name IEMOCAP \
  --bert_path ./pretrained/sup-simcse-roberta-large \
  --domain_anchor_path ./domain_anchors/IEMOCAP_M3_hyp.pt \
  --num_subanchors 3 \
  --anchor_dim 256 \
  --anchor_temperature 0.1 \
  --proto_temperature 0.1 \
  --anchor_logit_weight 0.3 \
  --proto_loss_weight 0.5 \
  --compact_loss_weight 0.1 \
  --ot_epsilon 0.05 \
  --ot_iters 50 \
  --prototype_momentum 0.9
```

---

## 23. 训练日志必须输出

每个 epoch 输出：

```text
loss_total
loss_ce
loss_proto
loss_compact
dev weighted F1
test weighted F1
```

还要输出锚点统计：

```text
same_anchor_sim_mean
same_anchor_sim_max
diff_anchor_sim_mean
diff_anchor_sim_max
```

还要输出 OT 分配统计：

```text
class angry assignment: [xx, xx, xx]
class excited assignment: [xx, xx, xx]
class frustrated assignment: [xx, xx, xx]
class happy assignment: [xx, xx, xx]
class neutral assignment: [xx, xx, xx]
class sad assignment: [xx, xx, xx]
```

这些统计用于判断：

```text
1. 异类锚点是否被分开。
2. 同类域子锚点是否塌缩。
3. 每个子锚点是否真的被样本使用。
4. OT 分配是否失衡。
```

---

## 24. 预期改进点

相比当前第一次训练版本，本模型改进了以下问题：

### 24.1 用 OT 替代最近邻 hard assignment

原来样本直接选择最近子锚点，容易导致所有样本挤到一个子锚点。现在通过 OT 分配，使每个子锚点都能获得一定样本质量，更符合 HMPEAE 的方法。

### 24.2 用聚类初始化保留域子锚点思想

HMPEAE 原型通常通过目标函数训练得到。本模型先用类别内聚类初始化，保证每个子锚点来自真实训练样本分布，保留“情感表达子域”的含义。

### 24.3 用超球面锚点优化保证类别边界

聚类得到的锚点可能仍然混在一起。因此引入 HMPEAE 风格的类间分离和语义排序，让不同情感类别的域子锚点在超球面上形成更清晰边界。

### 24.4 用紧致损失增强子域内部聚合

OT 分配后，样本会向对应子锚点靠近，从而让同一表达子域内部更紧凑。

### 24.5 用 EMA 保持锚点动态适应

训练过程中，锚点根据样本分配结果更新，而不是固定不变。这样域子锚点可以随着 ERC 表示空间调整。

---

## 25. 最终论文表述

可以写成：

```text
为缓解对话情感识别中同一情感类别内部表达差异较大以及不同情感类别边界模糊的问题，本文提出一种基于超球面域子锚点与最优传输的对话情感识别模型。首先，模型利用预训练语言模型编码带上下文的目标话语，并通过映射层获得低维情感表示。随后，在每个情感类别内部对训练样本表示进行聚类，以聚类中心初始化该类别的多个域子锚点，从而刻画同一情感类别下的不同表达子域。进一步地，参考超球面多原型方法，模型将域子锚点归一化到超球面空间中，并通过类间锚点分离、类内域约束和情感标签语义排序损失优化锚点分布。训练阶段，模型将样本与其真实情感类别下多个域子锚点之间的匹配建模为最优传输问题，并利用 Sinkhorn 算法获得软分配结果。在此基础上，模型通过情感分类损失、子锚点分类损失和样本—子锚点紧致损失联合优化话语表示，同时根据最优传输分配结果使用 EMA 动态更新域子锚点。最终，模型能够在保持情感类别内部子域差异的同时，增强不同情感类别之间的判别边界。
```

---

## 26. Codex 实现顺序

请 Codex 严格按下面顺序实现：

```text
1. 新增 generate_domain_anchors.py：
   从训练好的 encoder 中提取 z_i，按类别 KMeans，保存 [C,M,D] 域子锚点。

2. 新增 pretrain_hyp_domain_anchors.py：
   读取 KMeans 初始化锚点，使用 L_inter + L_domain + L_rank 优化锚点分布。

3. 新增 sinkhorn.py：
   实现 Sinkhorn-Knopp OT 分配。

4. 新增 hdsa_losses.py：
   实现 soft_cross_entropy、anchor_inter_loss、anchor_domain_loss、anchor_rank_loss、compactness loss。

5. 新增 hdsa_erc_model.py：
   实现 encoder、map_function、classifier、domain_anchors buffer、anchor logits、logits fusion。

6. 修改 trainer.py：
   当 args.use_hdsa=True 时，走 HDSA 训练逻辑：
   CE + proto CE + compact loss + EMA anchor update。

7. 修改 run.py：
   增加 HDSA 参数，并能选择 HDSA 模型。

8. 增加日志：
   输出 loss 分项、anchor 相似度统计、OT assignment counts、dev/test weighted F1。

9. 保证旧模型不受影响：
   args.use_hdsa=False 时，原训练路径保持不变。
```

---

## 27. 验收标准

实现完成后必须满足：

```text
1. 能成功生成 KMeans 域子锚点文件。
2. 能成功预训练超球面域子锚点文件。
3. HDSA 模型能正常加载预训练锚点。
4. Sinkhorn OT 分配不会出现 NaN。
5. 主训练能正常跑完至少 1 个 epoch。
6. 日志中能看到 loss_ce、loss_proto、loss_compact。
7. 日志中能看到 same/diff anchor similarity。
8. 日志中能看到每类子锚点分配数量。
9. 不使用 args.use_hdsa 时，原 EACL 模型仍能正常运行。
```

---

## 28. 最终一句话总结

```text
HDSA-ERC 先通过类别内聚类得到数据驱动的情感域子锚点，再参考 HMPEAE 将这些锚点放入超球面空间进行类间分离，并在 ERC 训练中使用最优传输完成样本到域子锚点的软分配，最终通过情感分类、子锚点分类和紧致约束共同提升对话情感识别效果。
```
