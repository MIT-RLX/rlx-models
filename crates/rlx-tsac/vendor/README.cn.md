# tsac-ng — 神经音频编解码器（多后端）

**tsac-ng v0.1.4** — 对 TSAC 神经音频编解码器的逆向工程重建。兼容 `.txc` 容器格式和 `.bin` 模型文件。

> **🤖 AI 辅助开发**：单人开发者配合 AI 编程助手完成 **102 轮调查**（R079-R180，4 个阶段）。
> 架构设计、GDB/objdump/LD_PRELOAD 地面真值提取和验证由人类主导；实现由 AI 增强。
> 详见 [METHODOLOGY.md](.ai/METHODOLOGY.md)。

---

## 兼容性状态

| 功能 | 状态 | % | 说明 |
|---------|:------:|:--:|-------|
| **自有快速 TXC 编解码** | ✅ | 100% | 原始 uint8 格式 |
| **原版快速 TXC 解码** | 🎯 | 90% | 索引 100%。**RMS 0.2023≈0.2029**。AVX-512 已修复 |
| **原版普通 TXC 解码** | 🔧 | 60% | Transformer+range coder 已实现。端到端待集成 |
| **CRC32 校验** | ✅ | 100% | 多项式 0x04C11DB7 |
| **详细输出** | ✅ | 100% | 全部匹配原版 |
| **DAC 解码架构** | ✅ | 95% | 32conv/29snake/4convtr GDB 验证 |
| **BF8 反量化** | ✅ | 80% | 全管线逆向。权重 corr 0.82 |
| **CPU SIMD** | ✅ | 95% | AVX-512/AVX2/NEON/SVE/RVV。AVX-512 bug 已修复 |
| **CUDA** | ✅ | 85% | 完整解码+编码 |
| **HIP** | ✅ | 65% | 编译通过 |
| **Vulkan** | 🔧 | 40% | 管线基础完成，解码/编码未接线 |
| **LLVM JIT** | 🔧 | 35% | 4 JIT 函数工作，解码图未完成 |
| **编码器** | ✅ | 70% | 跨步卷积已修复 |
| **Transformer** | ✅ | 80% | 12L GPT-2 已实现 |
| **Range Coder** | ✅ | 80% | get_freq+累积+多位 |
| **Convt 权重** | ✅ | 100% | GDB 确认 stride=K/2 |

### 🎯 Phase 4 里程碑

- **RMS 0.2023 ≈ 目标 0.2029** (99.7% 匹配)
- AVX-512 70× 放大 bug **已修复** (R161-R164)
- weight_g 仅用于 model.6 → RMS 精确匹配
- 0% 削波，多文件全通过
- 残留：WAV 相关性 ~0（BF8 权重 29% 残差）

---

## 快速开始

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
./tsac-ng -v d input.txc output.wav                       # 快速 TXC 解码
./tsac-ng -m /usr/share/tsac/dac_stereo_q8.bin d in.txc out.wav  # 原版解码
```

## 后端状态

| 后端 | 构建 | 运行时 | 备注 |
|---------|:-----:|:-------:|-------|
| CPU (x86-64) | ✅ | ✅ | AVX/AVX2/AVX-512 |
| CPU (ARM64) | ✅ | ✅ | NEON + SVE |
| CPU (RISC-V) | ✅ | ✅ | RVV + scalar |
| CUDA | ✅ | ✅ | SM 8.0+ |
| HIP/ROCm | ✅ | ✅ | gfx1030+ |
| Vulkan | ✅ | ⚠️ | 跨平台 |
| LLVM JIT | ✅ | ⚠️ | 实验性 |

## 已知限制

- 原版快速 TXC：RMS 已匹配，WAV 波形未对齐（BF8 权重残差）
- 普通 TXC：端到端集成待完成
- GPU：仅 CUDA 完整可用

## 许可证

MIT

---

```
tsac-ng v0.1.4 — Copyright (c) 2026 Hope2333 (幽零小喵)
```
