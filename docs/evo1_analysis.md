# Evo-1 代码库深度分析

> 目的：记录 Evo-1（`D:\Github_Project\Evo1_Project\Evo-1\Evo-1\`）的架构与实现细节，
> 作为 DualTreeVLA 对标 / 改进的基准。
> 每次发现新差异请在本文件末尾追加。

---

## 1. 目录结构

```
Evo_1/
├── model/
│   ├── action_head/
│   │   └── flow_matching.py       ← FlowmatchingActionHead（核心）
│   └── internvl3/
│       └── internvl3_embedder.py  ← InternVL3 VLM 嵌入器
├── dataset/
│   ├── config.yaml                ← 数据集配置（embodiment/path/view_map）
│   └── lerobot_dataset_pretrain_mp.py  ← LeRobotDataset（多进程 parquet cache）
├── scripts/
│   ├── Evo1.py                    ← EVO1 模型主类
│   └── train.py                   ← 训练入口
LIBERO_evaluation/
└── libero_client_4tasks.py        ← LIBERO 仿真评估（WebSocket client）
Evo1_LIBERO/
└── norm_stats.json                ← LIBERO 实际使用的 min/max 归一化 stats
```

---

## 2. 模型架构

### 2.1 EVO1 模型 (`Evo1.py`)

```
输入图像 (PIL) + 文本 prompt
       ↓
InternVL3Embedder.get_fused_image_text_embedding_from_tensor_images()
       ↓  完整 token 序列 (B, N_tokens, 896)，约 196+ image tokens + text tokens
       ↓  N_tokens 随输入变化，不固定
FlowmatchingActionHead.get_action(fused_tokens, state)
       ↓
(B, H_a × d_a)  → reshape → (H_a, d_a)
```

### 2.2 FlowmatchingActionHead (`flow_matching.py`)

**关键设计**：
- `embed_dim = 896`（与 InternVL3 隐层维度一致）
- `hidden_dim = 1024`
- `num_layers = 8`（深）
- `num_heads = 8`
- `horizon = 16`，`per_action_dim = 7`
- `num_inference_timesteps = 50`（默认，config可改为20）

**动作编码**（`MultiEmbodimentActionEncoder`）：
- `W1/W2/W3`：3层 MLP 把每步 action 投影到 `embed_dim`
- 加 Sinusoidal 位置编码

**Transformer Block**（`BasicTransformerBlock`）：
```python
# 注意：是 self-attention + cross-attention 分开的 Evo-1 风格
attn_out = self.attn(norm1(action_tokens), context_tokens, context_tokens)
  # Q = action, K = V = context_tokens（即 VLM 完整 token 序列）
x = action_tokens + attn_out
ff_out = self.ff(norm2(x) + time_emb)
x = x + ff_out
```

**时间戳嵌入**：Sinusoidal positional encoding，`time_index = int(t * 1000)`

**前向（训练）**：
```python
# 噪声采样：Uniform[-1, 1]
noise = torch.rand_like(actions_gt) * 2 - 1
# 时间：Beta(2,2) clamp(0.02, 0.98)
t = Beta(2,2).sample().clamp(0.02,0.98)
# 插值
action_intermediate = (1 - t) * noise + t * actions_gt
# 预测速度
pred_velocity = model(action_intermediate, t, context_tokens)
# 目标速度
target_velocity = (actions_gt - noise).view(B, -1)
# Loss：MSE，乘以 action_mask 补偿缩放
loss = MSE(pred_velocity * action_mask, target_velocity) * scale_factor
```

**前向（推理，`get_action`）**：
```python
action = rand * 2 - 1               # 初始 Uniform 噪声
for i in range(N=50):
    t = i / N
    pred = model(action, t, context)
    action = action + dt * pred      # Euler 积分
```

---

## 3. 数据集

### 3.1 `LeRobotDataset` (`lerobot_dataset_pretrain_mp.py`)

- **格式**：LeRobot v2 parquet + mp4 视频，通过 `view_map` 配置
- **多线程预处理**：第一次运行 pickle cache，后续直接读 `.pkl`
- **归一化**：`min-max → [-1, 1]`（不是 z-score）
  ```python
  state = 2 * (state - state_min) / (state_max - state_min + 1e-8) - 1
  action = 2 * (action - action_min) / (action_max - action_min + 1e-8) - 1
  ```
- **Action masking**：`action_mask`，对 padding 维度为 0
  - LIBERO 7 维，但 `max_action_dim=24`，所以 mask = `[1]*7 + [0]*17`
  - 同理 `state_mask` = `[1]*7 + [0]*17`（`max_state_dim=24`）
- **多 embodiment**：`arm_to_embodiment_id` 字典，传给 `CategorySpecificLinear`

### 3.2 归一化 stats 来源

存在 `Evo1_LIBERO/norm_stats.json`，结构：
```json
{
  "observation.state": {"min": [...], "max": [...]},
  "action": {"min": [...], "max": [...]}
}
```

---

## 4. LIBERO 评估 (`libero_client_4tasks.py`)

### 图像观测处理

```python
img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
```
**注意**：同时翻转行和列（`[::-1, ::-1]` = 旋转180°），**不是只翻转行**。
即 `np.rot90(img, 2)` 等效。

wrist image 同理：`obs["robot0_eye_in_hand_image"][::-1, ::-1]`

### state 构造

```python
state = np.concatenate((
    obs["robot0_eef_pos"],       # (3,)
    quat2axisangle(obs["robot0_eef_quat"]),  # (3,)
    obs["robot0_gripper_qpos"],  # (2,)
))  # → (8,)
```

### quat2axisangle（Evo-1 原版）

```python
def quat2axisangle(quat):
    if quat[3] > 1.0:  quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
```
**关键**：`acos(quat[3])` 不取 abs，与 DualTreeVLA 的修复一致。

### Gripper 二值化（Evo-1）

```python
if action[6] > 0.5:
    action[6] = -1    # ← > 0.5 → close (-1)
else:
    action[6] = 1     # ← ≤ 0.5 → open (+1)
```
**注意**：Evo-1 的预测值在 [-1,1] 空间（min-max 归一化）。与 DualTreeVLA 的 z-score 空间不同，阈值 0 vs 0.5 含义不同。

### action_mask（推理）

```python
"action_mask": [1] * 7 + [0] * 17   # max_action_dim=24
```
在 `get_action` 中：
```python
action_seq = action_seq * action_mask  # 每步前 mask 无效维度
```

---

## 5. 训练关键细节

- **lr = 1e-5**，`weight_decay = 1e-5`
- **max_steps=600**（小数据集），warmup_steps=300（warmup 占 50%！）
- **batch_size=16**，8GPU × bs=16 = 128 effective
- **VLM 微调**：默认 `--finetune_vlm` 解冻 InternVL3 部分层（`set_finetune_flags()`）
- **loss scaling**：`loss * (action_mask.numel() / (action_mask.sum() + 1e-8))`，补偿 mask 后的缩放
- **save checkpoint**：仅在 `step > 1000` 后才保存 best（防止早期 loss 震荡保存的坏 ckpt）

---

## 6. 与 DualTreeVLA 的关键差异（致命问题）

详见 `docs/project_status.md`。

| 维度 | Evo-1 | DualTreeVLA |
|---|---|---|
| Context tokens | ~200+ (VLM full sequence) | **1** (CrossModalFusion scalar) |
| Action head embed_dim | 896 | 256 |
| Action head layers | 8 | 4 |
| 归一化方式 | min-max → [-1,1] | z-score (mean/std) |
| State dim | 7 (masked, max=24) | 84 (8 valid, 76 zeros) |
| 图像旋转 | rot180 ([::-1,::-1]) | flipud only ([::-1]) |
| Gripper 阈值 | 0.5 | 0 |
| Noise 分布 | Uniform[-1,1] | Normal(0,1) |
| 训练步数 | max_steps 按需 | epochs (30-50) |
