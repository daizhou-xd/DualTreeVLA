# DualTreeVLA 项目状态 & 修复记录

> 每次对话修改后请在本文件末尾追加修改记录。
> 对话压缩后可直接阅读此文件恢复上下文。

---

## 1. 项目基本信息

- **服务器路径**：`/data/wyx/zd/DualTreeVLA`
- **conda 环境**：`dualtree`
- **硬件**：8× RTX A6000
- **基架**：在 Evo-1 (InternVL3+FlowMatching) 基础上，加入 DualMemoryTree + SGMTS 视觉模块
- **Evo-1 参考路径**：`D:\Github_Project\Evo1_Project\Evo-1\Evo-1\`

---

## 2. 模型架构

```
图像 (B,3,224,224)  ─── CLIP 归一化(mean/std) ──→
    ↓ CLIPPatchExtractor (frozen CLIP ViT-B/16)
    → patch 特征 (B, 196, 256)
    ↓ SGMTS (Semantic-Guided Mamba Tree Scan)
    → Z_v (B, 196, 256)   ← 保留全部 patch tokens
    → z_v_mean (B, 256)   ← 仅用于 fusion 和 HMT 更新

语言指令
    ↓ Qwen2.5-0.5B (LLM, hidden=896)
    → g_lang (B, 896)

HMT (Hierarchical Memory Tree)
    → TreeSSMReadout → Z_M (N_M, 256) → m_ctx = Z_M[-1] (B, 256)
    → MLPElevation → s_top (B, 256)

本体感知 state (B, 8)   ← d_q=8（LIBERO 有效维度）

CrossModalFusion(z_v_mean, m_ctx, g_lang, q)
    → f_fused (B, 1, 256)

cat([f_fused, Z_v], dim=1)
    → ctx (B, 197, 256)   ← 197 token = 1 fused + 196 patch

FlowMatchingActionHead(ctx)  ← cross-attn over 197 tokens
    → a_pred (B, 16, 7)
```

---

## 3. 问题列表（按严重程度）

### ✅ P0 - Context 只有 1 token（已修复 2026-04-11）

`CrossModalFusion` 输出 `(B, 1, 256)` → FlowBlock cross-attn 退化。

**修复**：`step()` 和 `forward()` 中把 `Z_v (B,196,256)` 与 `Z_fused (B,1,256)` 拼接，
得到 `ctx (B,197,256)` 再传入 action head。

**文件**：`dual_tree_vla/model/dual_tree_vla.py`

### ✅ P0 - CLIP 图像归一化缺失（已修复 2026-04-11）

**修复**：在 `CLIPPatchExtractor.__init__` 中注册 `clip_mean`/`clip_std` buffer，
`forward()` 开头对 `[0,1]` 图像做标准归一化后再送 CLIP ViT。

**文件**：`dual_tree_vla/model/sgmts/sgmts.py`

### ✅ P1 - State 维度浪费 8/84（已修复 2026-04-11）

**修复**：`configs/train_phase1.yaml` 和 `configs/train_phase2.yaml` 中
`data.d_q` 和 `model.d_q` 从 84 → 8，`init_from` 均置为 `null`（需从头训练）。

### P1 - 图像预处理：翻转方式与 Evo-1 不同

- Evo-1：`img[::-1, ::-1]`（旋转 180°，即 rot90×2）
- DualTreeVLA：`img[::-1]`（只翻转行，上下翻转）

需要核实训练数据集（lerobot 格式）存储的是哪种翻转，eval 必须与训练一致。

**当前状态**：`--no_image_flip` flag 可绕过，加 `--debug_first_ep` 观察效果。

### P2 - Gripper 二值化阈值

- Evo-1（min-max 归一化，open=1.0）：`action[6] > 0.5 → close(-1)`
- DualTreeVLA（z-score 归一化）：`action[6] > 0.0 → open(1.0)`

两者在各自空间内含义不同。若训练数据中 gripper 的均值不为 0，z-score 阈值 0 会偏移。

---

## 4. 已修复问题历史

| 日期 | 问题 | 修复文件 |
|---|---|---|
| 早期 | `llm_path` 错误 | `scripts/train.py` |
| 早期 | `DistributedDataParallelKwargs` 缺失 | `scripts/train.py` |
| 会话A | Invalid action WARN 刷屏 | `eval_libero_sim.py` - terminated flag |
| 会话A | 动作未反归一化 | `eval_libero_sim.py` - ActionNorm 类 |
| 会话A | state 未归一化 | `eval_libero_sim.py` - normalize_state() |
| 会话A | `quat2axisangle` abs(w) bug | `eval_libero_sim.py` - _quat2axisangle |
| 会话A | CLIP 未加载（主因 SR=0） | `eval_libero_sim.py` + `eval.py` - clip_model_name |
| 会话A | `past_key_values` 警告 | `dual_tree_vla/model/dual_tree_vla.py` - use_cache=False |
| 会话B | 视频无法保存（imageio缺失） | `eval_libero_sim.py` - cv2 fallback |
| 会话B | debug 只打印第0步 | `eval_libero_sim.py` - debug_steps 参数 |
| 会话B | 训练无可视化 | `scripts/train.py` - visualize_epoch() |
| 2026-04-11 | ✅ P0 Context=1 token | `dual_tree_vla.py` - cat([Z_fused, Z_v]) → ctx (B,197,256) |
| 2026-04-11 | ✅ P0 CLIP 归一化缺失 | `sgmts.py` - register_buffer clip_mean/std, 归一化 forward |
| 2026-04-11 | ✅ P1 d_q=84→8 | `train_phase1.yaml` + `train_phase2.yaml` - d_q 改 8, init_from=null |
| 2026-04-11 | 新增 eval_server.py | `scripts/eval_server.py` - WebSocket 推理服务端 |
| 2026-04-11 | 新增 eval_client.py | `scripts/eval_client.py` - LIBERO 仿真 WebSocket 客户端 |

---

## 5. 当前评估状态

### eval_libero_sim.py debug 输出（phase2 当前 ckpt）

```
[DEBUG step 0] action norm: [ 1.64  1.76  0.93  0.04 -1.24  0.89  0.27]
               action denorm: [ 0.48  0.69  0.29  0.01 -0.06  0.07  0.17]
...
均值 norm[dim=0] ≈ +1.8（持续偏正 x 方向）
```

**结论**：动作头输出与观测无关，向固定方向偏置 → Context=1 token 的直接结果。

### SR（Success Rate）

- LIBERO-10 Phase2 当前 ckpt：**0%**
- 根本原因：Context=1 token 导致 flow matching 无法条件化

---

## 6. 文件索引（关键文件）

| 文件 | 用途 | 重要程度 |
|---|---|---|
| `dual_tree_vla/model/dual_tree_vla.py` | 主模型，`step()` 推理入口 | ★★★★★ |
| `dual_tree_vla/model/action_head/flow_matching.py` | DiT风格FlowBlock | ★★★★★ |
| `dual_tree_vla/model/fusion.py` | CrossModalFusion（P0 问题所在） | ★★★★★ |
| `dual_tree_vla/model/sgmts/sgmts.py` | SGMTS + CLIP patch extractor | ★★★★ |
| `dual_tree_vla/dataset/libero.py` | 训练数据加载，z-score 归一化 | ★★★★ |
| `scripts/eval_libero_sim.py` | 仿真评估（单进程，含 ActionNorm） | ★★★★ |
| `scripts/eval_server.py` | WebSocket 推理服务端（GPU 节点） | ★★★★ |
| `scripts/eval_client.py` | WebSocket LIBERO 仿真客户端 | ★★★★ |
| `scripts/train.py` | Phase1/2 训练，含 visualize_epoch | ★★★ |
| `configs/train_phase2.yaml` | Phase2 训练配置 | ★★★ |
| `docs/evo1_analysis.md` | Evo-1 代码分析参考 | ★★★★ |

---

## 7. 环境 & 工具

```bash
# 归一化 stats（z-score，mean/std）
dataset/datasets/libero_10/meta/stats.json

# 训练可视化视频（每 epoch 输出）
results/viz/phase2/phase2_ep001.mp4 ...

# 仿真评估视频（debug 时输出）
results/videos/libero_10/debug_task00_ep00_fail.mp4
results/videos/libero_10/fail_task00_ep01.mp4

# imageio 未安装，VideoWriter 使用 cv2 fallback
```

---

## 8. 待办（下一步）

1. **重新训练 Phase 1**（d_q=8 导致架构变化，旧 ckpt 不兼容）：
   ```bash
   bash scripts/train_phase1.sh
   ```
2. **训练 Phase 2**：
   ```bash
   bash scripts/train_phase2.sh
   ```
3. **服务端/客户端评估**：
   ```bash
   # 终端1（GPU 节点）
   python scripts/eval_server.py --ckpt checkpoints/runs/phase2/phase2_best.pt
   # 终端2
   python scripts/eval_client.py --server ws://127.0.0.1:9000 --suite libero_10
   ```
4. 观察 `visualize_epoch` 视频确认 MAE < 0.1 后再启动仿真评估。

---

## 9. 修改记录

### 2026-04-11

- 创建 `scripts/eval_libero_sim.py` 仿真评估脚本
- 修复 CLIP 未加载（最主要 SR=0 原因）
- 添加 `--debug_first_ep`, `--record_fail`, `--no_image_flip` 参数
- `_save_video` 添加 cv2 fallback
- `_debug_step` 改为打印前 N 步并输出整集统计
- `scripts/train.py` 添加 `visualize_epoch()`：每个 epoch 保存 GT vs Pred 对比视频
- 更新 README.md 训练命令 + 可视化章节
- 创建 `docs/evo1_analysis.md` + `docs/project_status.md`

### 2026-04-11（续）

- **Fix 1 (P0)**：`dual_tree_vla/model/dual_tree_vla.py`
  - `step()` 和 `forward()` 中：`ctx = cat([Z_fused(B,1,256), Z_v(B,196,256)], dim=1)` → `(B,197,256)`
  - action head cross-attn 现在可 attend 197 个 token
- **Fix 2 (P0)**：`dual_tree_vla/model/sgmts/sgmts.py`
  - `CLIPPatchExtractor.__init__` 注册 `clip_mean`/`clip_std` buffer
  - `forward()` 中：`x = (x.float() - clip_mean) / clip_std` 后再送 CLIP ViT
- **Fix 3 (P1)**：`configs/train_phase1.yaml`, `configs/train_phase2.yaml`
  - `data.d_q`: 84 → 8；`model.d_q`: 84 → 8
  - `init_from` / `pretrain_ckpt` 置为 `null`（旧权重架构不兼容）
- **新增 `scripts/eval_server.py`**：WebSocket 推理服务端
  - 加载 DualTreeVLA + ActionNorm，监听 WebSocket
  - `reset` 消息 → `model.reset_trees()`
  - `infer` 消息 → `model.step()` + 去归一化 → 返回 `{"actions": [[7f]×H_a]}`
- **新增 `scripts/eval_client.py`**：LIBERO 仿真 WebSocket 客户端
  - 驱动 LIBERO MuJoCo 环境，支持 `--suites` 评估多个 suite
  - 每 episode：send reset → 热身10步 → 循环(infer+执行 horizon 步)
  - 支持 `--save_video`, `--out` JSON, `--gripper_thresh`
- **`requirements.txt`**：新增 `websockets>=12.0`
