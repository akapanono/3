# ERC 模型完整设计：基于预编码聚类域子锚点与超球面锚点分离的对话情感识别模型

## 1. 模型核心思想

本模型用于对话情感识别任务（Emotion Recognition in Conversation, ERC）。  
它的核心目标是解决两个问题：

1. **同一情感类别内部表达差异较大**  
   例如 `angry` 可能包含指责型愤怒、抱怨型愤怒、压抑型愤怒；`sad` 可能包含直接悲伤、失落、无助等不同表达方式。  
   如果每个情感类别只用一个锚点表示，会压缩类别内部差异。

2. **不同情感类别之间边界模糊**  
   ERC 中常见的混淆类别包括：
   - `angry` 与 `frustrated`
   - `happy` 与 `excited`
   - `sad` 与 `neutral`
   - `neutral` 与 `frustrated`

因此，本模型不再使用单一情感锚点，而是为每个情感类别构建多个 **域子锚点**。  
这些域子锚点不是手工指定的，而是先通过预编码器提取训练样本表示，再在每个情感类别内部聚类得到。  
之后，模型在训练过程中让样本靠近本类对应的子锚点，同时把不同情感类别的子锚点拉远，从而形成更加清晰的情感表示空间。

模型整体可以概括为：

```text
对话输入
  → 预编码器提取话语表示
  → 按情感类别聚类得到域子锚点
  → 将样本表示和子锚点映射到超球面空间
  → 样本靠近本类子锚点
  → 不同情感类别子锚点相互拉远
  → 分类头与锚点相似度共同完成情感预测
```

---

## 2. 模型名称

建议模型中文名称：

```text
基于聚类域子锚点与超球面分离的对话情感识别模型
```

建议英文名称：

```text
Cluster-guided Hyperspherical Domain Sub-anchor Network for ERC
```

简称：

```text
CHDS-ERC
```

其中：

```text
C：Cluster-guided，表示子锚点由聚类引导得到
H：Hyperspherical，表示使用超球面表示空间
D：Domain，表示同一情感类别内部的表达子域
S：Sub-anchor，表示每个情感类别包含多个子锚点
ERC：Emotion Recognition in Conversation
```

---

## 3. 任务定义

给定一段对话：

```text
D = {u_1, u_2, ..., u_T}
```

其中每个话语包含说话人和文本：

```text
u_i = (speaker_i, text_i)
```

模型需要预测目标话语 `u_t` 的情感类别：

```text
y_t ∈ Y
```

以 IEMOCAP 为例，情感类别集合为：

```text
Y = {happy, sad, neutral, angry, excited, frustrated}
```

模型输入不仅包括当前目标话语，还包括目标话语之前的上下文。

---

## 4. 整体模型流程

整个模型只保留一条主线，不拆分成多个版本。  
完整流程如下：

```text
Step 1：构造对话上下文 prompt
Step 2：使用预编码器提取目标话语表示
Step 3：使用映射层得到低维情感表示
Step 4：对训练集表示按类别聚类，得到每类多个域子锚点
Step 5：将样本表示和域子锚点归一化到超球面空间
Step 6：计算样本与所有子锚点的相似度
Step 7：样本向本类最匹配子锚点靠近
Step 8：不同情感类别的子锚点通过 hard negative 方式拉远
Step 9：同类子锚点保持适度差异，避免完全塌缩
Step 10：融合分类头与锚点相似度，完成最终情感预测
```

---

## 5. 输入与 Prompt 构造

### 5.1 对话上下文格式

对于一段对话，先把历史话语转化为统一格式：

```text
{speaker} says: {utterance_text}
```

例如：

```text
A says: I cannot believe you did that.
B says: I already apologized.
A says: That is not enough.
```

### 5.2 目标话语 prompt

对目标话语构造情感提示：

```text
For utterance: {target_text} {speaker} feels <mask>
```

例如目标话语是：

```text
I already apologized.
```

说话人是：

```text
B
```

则目标 prompt 为：

```text
For utterance: I already apologized. B feels <mask>
```

### 5.3 最终输入

最终输入由历史上下文和目标 prompt 拼接得到：

```text
A says: I cannot believe you did that.
B says: I already apologized.
A says: That is not enough.
For utterance: I already apologized. B feels <mask>
```

模型通过 `<mask>` 位置的 hidden state 表示目标话语的情感语义。

---

## 6. 预编码器：提取目标话语表示

### 6.1 编码器选择

使用预训练语言模型作为对话编码器：

```text
Sup-SimCSE-RoBERTa-large
```

也可以使用：

```text
RoBERTa-large
```

设编码器为：

```text
Encoder(·)
```

输入 prompt 后得到 token-level hidden states：

```text
H = Encoder(X)
```

其中：

```text
H ∈ R^{L × d}
```

`L` 是输入序列长度，`d` 是隐藏层维度。  
如果使用 RoBERTa-large，则：

```text
d = 1024
```

### 6.2 取 `<mask>` 表示

找到 `<mask>` token 的位置，取该位置 hidden state 作为目标话语表示：

```text
h_i = H[mask_pos]
```

其中：

```text
h_i ∈ R^d
```

`h_i` 是当前目标话语的上下文感知表示。

---

## 7. 映射层：进入情感锚点空间

直接使用 PLM 输出的 `h_i` 维度较高，且未必适合锚点对比学习。  
因此需要一个映射层，将其映射到低维情感空间：

```text
z_i = f_map(h_i)
```

映射层结构为：

```python
map_function = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, anchor_dim)
)
```

其中：

```text
hidden_dim = 1024
anchor_dim = 256 或 512
```

映射后进行 L2 归一化：

```text
z_i = normalize(z_i)
```

此时：

```text
z_i ∈ R^{anchor_dim}
```

`z_i` 是后续聚类、锚点匹配、对比学习和分类的核心表示。

---

## 8. 预编码聚类：生成域子锚点

### 8.1 为什么要聚类定锚点

原始 emotion anchor 通常来自人工模板，例如：

```text
The speaker feels angry.
The speaker feels sad.
The speaker feels happy.
```

这种方法可以提供情感语义先验，但它的问题是：  
模板锚点不一定能反映训练集中真实话语的分布。

因此，本模型采用 **预编码聚类定锚点**：

```text
先使用编码器提取训练集中所有样本的情感表示 z_i，
再在每个情感类别内部进行聚类，
每个聚类中心作为该情感类别的一个域子锚点。
```

这样得到的子锚点来自真实样本分布，而不是完全依赖人工模板。

---

### 8.2 类别内聚类

设训练集为：

```text
{(x_i, y_i)}_{i=1}^{N}
```

先通过预编码器和映射层得到所有训练样本表示：

```text
z_i = f_map(Encoder(x_i)[mask_pos])
```

对于每个情感类别 `c`，收集该类别所有样本表示：

```text
Z_c = {z_i | y_i = c}
```

然后在 `Z_c` 内部进行聚类：

```text
KMeans(Z_c, K) → {a_{c,1}, a_{c,2}, ..., a_{c,K}}
```

其中：

```text
K = 每个情感类别的子锚点数量
a_{c,k} = 第 c 类情感的第 k 个域子锚点
```

所有类别的子锚点组成：

```text
A ∈ R^{C × K × D}
```

其中：

```text
C = 情感类别数
K = 每类子锚点数
D = anchor_dim
```

---

### 8.3 子锚点含义

以 `angry` 为例，聚类后可能形成：

```text
a_angry,1：指责型愤怒
a_angry,2：抱怨型愤怒
a_angry,3：压抑型愤怒
```

以 `sad` 为例，可能形成：

```text
a_sad,1：直接悲伤
a_sad,2：失落无助
a_sad,3：低落沉默
```

这些含义不是人工标注出来的，而是由类别内部样本分布自动形成。  
模型不需要显式知道每个子锚点对应什么语义，只需要利用这些子锚点刻画类内差异。

---

## 9. 超球面空间：统一样本和锚点表示

聚类得到子锚点后，需要把样本表示和子锚点放到同一个超球面空间中。

对样本表示归一化：

```text
z_i = normalize(z_i)
```

对子锚点归一化：

```text
a_{c,k} = normalize(a_{c,k})
```

这样所有向量都在单位超球面上。  
后续距离和相似度都用余弦相似度计算：

```text
sim(z_i, a_{c,k}) = cos(z_i, a_{c,k})
```

距离定义为：

```text
D(z_i, a_{c,k}) = 1 - cos(z_i, a_{c,k})
```

使用超球面空间的好处是：

```text
1. 向量长度不会干扰距离计算。
2. 类别之间主要通过角度区分。
3. 更适合进行原型 / 锚点分离。
```

---

## 10. 样本与子锚点匹配

对于一个训练样本 `x_i`，其真实标签为：

```text
y_i
```

模型只在该真实类别的 `K` 个子锚点中寻找最匹配的一个。

计算：

```text
s_{i,k} = cos(z_i, a_{y_i,k})
```

选择最相似的子锚点：

```text
k* = argmax_k s_{i,k}
```

该样本的正锚点为：

```text
a_i^+ = a_{y_i,k*}
```

这一步的作用是：  
同一情感类别内部，不同表达方式的样本可以自动靠近不同的子锚点。

例如：

```text
angry 中的指责型样本 → 靠近 angry 的第一个子锚点
angry 中的压抑型样本 → 靠近 angry 的第二个子锚点
angry 中的抱怨型样本 → 靠近 angry 的第三个子锚点
```

---

## 11. 样本—子锚点紧致约束

为了让样本真正聚集到对应子锚点附近，引入样本—子锚点紧致损失：

```text
L_pull = mean_i [1 - cos(z_i, a_i^+)]
```

其中：

```text
z_i = 样本表示
a_i^+ = 该样本所属类别中最匹配的子锚点
```

这个损失的作用是：

```text
同一类别内部的样本不是全部挤到一个中心，
而是根据表达差异分别靠近不同的域子锚点。
```

代码形式：

```python
def anchor_pull_loss(reps, labels, anchors):
    reps = F.normalize(reps, dim=-1)
    anchors = F.normalize(anchors, dim=-1)

    cur_anchors = anchors[labels]      # [B, K, D]
    sim = torch.einsum("bd,bkd->bk", reps, cur_anchors)

    best_k = sim.argmax(dim=1)
    batch_idx = torch.arange(reps.size(0), device=reps.device)
    pos_anchor = cur_anchors[batch_idx, best_k]

    loss = 1.0 - F.cosine_similarity(reps, pos_anchor, dim=-1)
    return loss.mean()
```

---

## 12. 不同类别子锚点分离

### 12.1 为什么要分离锚点

仅让样本靠近本类子锚点还不够。  
如果不同情感类别的子锚点靠得太近，模型仍然会混淆。

例如：

```text
angry 的某个子锚点
```

可能和：

```text
frustrated 的某个子锚点
```

非常接近。  
这种情况下，模型遇到边界样本时就容易预测错误。

因此，需要显式拉远不同情感类别的子锚点。

---

### 12.2 Hard Negative 拉远策略

本模型参考超球面多原型分离思想，不是平均拉开所有异类子锚点，而是重点处理最容易混淆的异类子锚点。

对于每个子锚点：

```text
a_{c,k}
```

找到与它最相似的异类子锚点：

```text
a_{c',l}, 其中 c' ≠ c
```

也就是：

```text
max_{c'≠c,l} cos(a_{c,k}, a_{c',l})
```

如果这个最大相似度很高，说明存在一个异类子锚点与当前子锚点靠得太近，需要把它们拉远。

---

### 12.3 异类子锚点分离损失

定义：

```text
L_inter = mean_{c,k} max_{c'≠c,l} cos(a_{c,k}, a_{c',l})
```

训练时最小化 `L_inter`。  
这会降低每个子锚点与最近异类子锚点之间的余弦相似度。

直观理解：

```text
每个子锚点都检查自己最近的异类邻居是谁；
如果最近的异类邻居太近，就把它推远；
这样情感类别之间的边界会更加清晰。
```

代码形式：

```python
def hyperspherical_inter_anchor_loss(anchors):
    C, K, D = anchors.shape

    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(C * K, D)

    sim = torch.matmul(flat, flat.t())  # [C*K, C*K]

    labels = torch.arange(C, device=anchors.device).repeat_interleave(K)
    diff_mask = labels[:, None] != labels[None, :]

    hardest_diff_sim = sim.masked_fill(~diff_mask, -1e4).max(dim=1).values

    return hardest_diff_sim.mean()
```

---

## 13. 同类子锚点防塌缩约束

### 13.1 为什么需要防塌缩

每个情感类别有多个子锚点，但如果没有约束，它们可能会逐渐变得非常接近，甚至完全重合。  
一旦同类子锚点重合，多子锚点就失去了意义。

因此，需要防止同类子锚点塌缩。

---

### 13.2 只限制“过度相似”

同类子锚点仍然属于同一个情感类别，所以不能把它们强行拉得过远。  
本模型只限制它们不要过于相似。

设同一类别内两个子锚点为：

```text
a_{c,k}
a_{c,l}
```

如果：

```text
cos(a_{c,k}, a_{c,l}) > δ_same
```

说明它们太接近，需要惩罚。

损失定义：

```text
L_intra = mean ReLU(cos(a_{c,k}, a_{c,l}) - δ_same)
```

推荐：

```text
δ_same = 0.85
```

代码形式：

```python
def intra_anchor_diversity_loss(anchors, same_upper=0.85):
    C, K, D = anchors.shape

    if K <= 1:
        return anchors.new_tensor(0.0)

    anchors = F.normalize(anchors, dim=-1)
    flat = anchors.reshape(C * K, D)

    sim = torch.matmul(flat, flat.t())

    labels = torch.arange(C, device=anchors.device).repeat_interleave(K)
    same_mask = labels[:, None] == labels[None, :]

    eye = torch.eye(C * K, dtype=torch.bool, device=anchors.device)
    same_mask = same_mask & (~eye)

    same_sim = sim[same_mask]

    if same_sim.numel() == 0:
        return anchors.new_tensor(0.0)

    return F.relu(same_sim - same_upper).mean()
```

---

## 14. 锚点分类头

除了普通分类头，模型还使用锚点相似度进行分类。

### 14.1 计算样本与所有子锚点相似度

对于一个 batch 的样本表示：

```text
Z ∈ R^{B × D}
```

子锚点矩阵：

```text
A ∈ R^{C × K × D}
```

计算相似度：

```text
S = Z · A
```

代码：

```python
scores = torch.einsum("bd,ckd->bck", z, anchors)
```

得到：

```text
scores ∈ R^{B × C × K}
```

含义：

```text
每个样本与每个情感类别下每个子锚点的相似度。
```

---

### 14.2 子锚点聚合为类别分数

由于每个类别有 `K` 个子锚点，需要把 `[C, K]` 的子锚点分数聚合为 `[C]` 的类别分数。

使用 LogSumExp 聚合：

```text
logit_anchor(i,c) = τ · log Σ_k exp(scores(i,c,k) / τ)
```

其中 `τ` 是温度系数。

LogSumExp 的优点是：

```text
1. 比 max 更平滑。
2. 不会只让一个子锚点有梯度。
3. 适合多子锚点分类。
```

代码：

```python
def aggregate_anchor_scores(scores, temperature=0.1):
    return torch.logsumexp(scores / temperature, dim=-1) * temperature
```

得到：

```text
logits_anchor ∈ R^{B × C}
```

---

## 15. 普通分类头

模型同时保留普通分类头：

```text
logits_cls = Wz + b
```

代码：

```python
logits_cls = classifier(z)
```

其中：

```text
z = 映射后的目标话语表示
```

普通分类头的作用是：

```text
提供稳定的类别监督，避免模型完全依赖锚点空间。
```

---

## 16. 最终预测方式

最终 logits 由普通分类头和锚点分类头融合得到：

```text
logits = (1 - β) logits_cls + β logits_anchor
```

其中：

```text
β ∈ [0,1]
```

推荐：

```text
β = 0.5
```

如果希望更依赖锚点结构，可以增大 `β`；  
如果锚点还不稳定，可以减小 `β`。

最终预测：

```text
ŷ_i = argmax_c logits(i,c)
```

---

## 17. 监督对比学习

模型保留 supervised contrastive learning。

对 batch 中的样本表示：

```text
z_i
```

以及域子锚点：

```text
a_{c,k}
```

构造对比学习样本集合。

正样本：

```text
与 z_i 情感标签相同的样本表示
与 z_i 情感标签相同的子锚点
```

负样本：

```text
其他情感类别的样本表示
其他情感类别的子锚点
```

这个损失的作用是：

```text
进一步拉近同类样本与同类锚点，
推远异类样本与异类锚点。
```

记为：

```text
L_supcon
```

---

## 18. 总损失函数

模型最终损失由五部分组成：

```text
L_total =
    λ_ce    L_ce
  + λ_sup   L_supcon
  + λ_pull  L_pull
  + λ_inter L_inter
  + λ_intra L_intra
```

其中：

```text
L_ce：最终 logits 的交叉熵分类损失
L_supcon：监督对比学习损失
L_pull：样本向本类最匹配子锚点靠近
L_inter：不同情感类别子锚点分离
L_intra：同类子锚点防塌缩
```

推荐初始权重：

```text
λ_ce = 0.3
λ_sup = 0.7
λ_pull = 0.1
λ_inter = 0.05
λ_intra = 0.01
```

整体训练目标是：

```text
让样本表示具有分类能力；
让同类样本和同类锚点靠近；
让不同情感类别锚点之间形成清晰边界；
让同一类别内部多个子锚点保持不同表达子域。
```

---

## 19. 完整训练流程

整个训练过程是一条完整主线：

```text
1. 构造带上下文的 prompt 输入。
2. 使用预训练编码器提取 `<mask>` 表示 h_i。
3. 通过映射层得到情感空间表示 z_i。
4. 使用训练集 z_i，按情感类别进行类别内聚类。
5. 每个类别得到 K 个聚类中心，作为该类别的域子锚点。
6. 将样本表示和域子锚点归一化到超球面空间。
7. 计算样本与所有子锚点的余弦相似度。
8. 使用 LogSumExp 将每类多个子锚点分数聚合为类别分数。
9. 融合普通分类头 logits 和锚点分类 logits。
10. 使用 CE、SupCon、L_pull、L_inter、L_intra 联合训练。
11. 推理时输入对话 prompt，得到 z_i，与各类别子锚点计算相似度，并结合分类头得到最终情感类别。
```

---

## 20. 推理流程

推理时不再聚类。  
聚类只在训练前用于确定域子锚点。

推理过程：

```text
输入对话上下文和目标话语
  → 构造 prompt
  → 编码器提取 <mask> 表示
  → 映射层得到 z
  → 计算 z 与所有域子锚点的相似度
  → 聚合每个类别的子锚点相似度
  → 与普通分类头输出融合
  → 得到最终情感类别
```

公式：

```text
h = Encoder(x)[mask_pos]
z = normalize(f_map(h))
scores = cos(z, A)
logits_anchor = logsumexp(scores)
logits_cls = classifier(z)
logits = (1 - β) logits_cls + β logits_anchor
prediction = argmax(logits)
```

---

## 21. 模型结构伪代码

```python
class CHDSERCModel(nn.Module):
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

        self.anchors = nn.Parameter(
            load_cluster_anchors(args.cluster_anchor_path),
            requires_grad=True
        )

        self.temperature = args.temperature
        self.anchor_logit_weight = args.anchor_logit_weight

    def forward(self, input_ids, attention_mask, mask_pos):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden = outputs.last_hidden_state
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)

        h = hidden[batch_idx, mask_pos]

        z = self.map_function(h)
        z = F.normalize(z, dim=-1)

        anchors = F.normalize(self.anchors, dim=-1)

        logits_cls = self.classifier(z)

        scores = torch.einsum("bd,ckd->bck", z, anchors)
        logits_anchor = torch.logsumexp(
            scores / self.temperature,
            dim=-1
        ) * self.temperature

        logits = (
            (1 - self.anchor_logit_weight) * logits_cls
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

## 22. 训练伪代码

```python
for batch in train_loader:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    mask_pos = batch["mask_pos"]
    labels = batch["labels"]

    outputs = model(input_ids, attention_mask, mask_pos)

    logits = outputs["logits"]
    z = outputs["z"]
    anchors = outputs["anchors"]

    loss_ce = cross_entropy(logits, labels)

    loss_supcon = supcon_loss(
        reps=z,
        labels=labels,
        anchors=anchors
    )

    loss_pull = anchor_pull_loss(
        reps=z,
        labels=labels,
        anchors=anchors
    )

    loss_inter = hyperspherical_inter_anchor_loss(
        anchors=anchors
    )

    loss_intra = intra_anchor_diversity_loss(
        anchors=anchors,
        same_upper=0.85
    )

    loss = (
        lambda_ce * loss_ce
        + lambda_sup * loss_supcon
        + lambda_pull * loss_pull
        + lambda_inter * loss_inter
        + lambda_intra * loss_intra
    )

    loss.backward()
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

---

## 23. 模型需要保存的内容

训练完成后保存：

```text
1. encoder 参数
2. map_function 参数
3. classifier 参数
4. 域子锚点 anchors
5. 模型超参数
6. label2id / id2label
```

保存格式：

```python
{
    "model_state_dict": model.state_dict(),
    "args": args,
    "label2id": label2id,
    "id2label": id2label,
    "anchors": model.anchors.detach().cpu()
}
```

---

## 24. 需要记录的训练日志

为了判断模型是否真的有效，需要记录以下内容。

### 24.1 分类指标

```text
train loss
dev weighted F1
test weighted F1
accuracy
macro F1
```

### 24.2 损失分项

```text
L_ce
L_supcon
L_pull
L_inter
L_intra
```

### 24.3 锚点分离情况

```text
同类子锚点平均余弦相似度
同类子锚点最大余弦相似度
异类子锚点平均余弦相似度
异类子锚点最大余弦相似度
```

理想情况：

```text
异类子锚点最大余弦相似度逐渐下降；
同类子锚点不会全部接近 1；
样本到本类子锚点的距离逐渐下降。
```

### 24.4 子锚点分配数量

记录每个类别内部样本分配到各个子锚点的数量：

```text
angry: [120, 95, 108]
sad: [80, 76, 91]
neutral: [200, 188, 176]
...
```

如果出现：

```text
angry: [320, 0, 0]
```

说明该类别子锚点发生塌缩或分配不均。

---

## 25. 推荐超参数

以 IEMOCAP 为例：

```text
num_classes = 6
num_subanchors = 3
hidden_dim = 1024
anchor_dim = 256 或 512
temperature = 0.1
anchor_logit_weight = 0.5
dropout = 0.1
batch_size = 8
epochs = 8
plm_lr = 1e-5
other_lr = 4e-4
max_grad_norm = 5.0
```

损失权重：

```text
lambda_ce = 0.3
lambda_sup = 0.7
lambda_pull = 0.1
lambda_inter = 0.05
lambda_intra = 0.01
```

聚类设置：

```text
K = 3
聚类空间 = mapped representation z
聚类方式 = KMeans + center normalize
```

---

## 26. 模型创新点

本模型的创新点可以概括为三点：

### 26.1 聚类初始化域子锚点

模型不再完全依赖人工情感模板生成锚点，而是先用预编码器提取训练集话语表示，再在每个情感类别内部聚类，用聚类中心作为域子锚点。  
这样可以让每个情感类别内部形成多个真实数据驱动的表达子域。

### 26.2 超球面异类子锚点分离

模型将所有样本表示和子锚点归一化到超球面空间，并对每个子锚点寻找最接近的异类子锚点作为 hard negative，通过降低它们的余弦相似度来扩大不同情感类别的决策边界。

### 26.3 样本—子锚点紧致学习

模型让每个样本靠近其真实类别下最匹配的子锚点，使同一情感类别内部的不同表达方式形成更紧凑的子簇，从而增强表示空间的判别性。

---

## 27. 可以写进论文的方法描述

```text
本文提出一种基于聚类域子锚点与超球面分离的对话情感识别模型。首先，模型利用预训练语言模型编码带上下文的目标话语，并取提示模板中 <mask> 位置的隐藏状态作为目标话语表示。随后，经过映射层得到低维情感表示，并在训练集上按照情感类别进行类别内聚类，将每个聚类中心作为该情感类别的域子锚点，以刻画同一情感类别内部的不同表达子域。进一步地，模型将样本表示和域子锚点归一化到超球面空间中，通过样本—子锚点紧致损失使样本靠近其所属类别下最匹配的子锚点；同时，为增强不同情感类别之间的判别边界，模型对每个子锚点选取最相近的异类子锚点作为 hard negative，并通过异类子锚点分离损失降低二者余弦相似度。此外，模型引入同类子锚点防塌缩约束，避免同一情感类别内部多个子锚点退化为单一表示。最终，模型融合普通分类头输出与锚点相似度分类输出，并结合交叉熵损失、监督对比损失、样本—锚点紧致损失和超球面锚点分离损失进行联合优化。
```

---

## 28. 一句话总结模型

```text
本模型先通过预编码器提取对话话语表示，再在每个情感类别内部聚类得到多个域子锚点，随后将样本和锚点统一到超球面空间中，使样本靠近本类子锚点、不同类别子锚点相互远离，最终通过分类头和锚点相似度共同完成对话情感识别。
```
