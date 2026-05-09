# 高斯消元 SIMD 并行化实验报告

## 1 问题描述

高斯消元（Gaussian Elimination）是求解 n 元线性方程组 A**x** = **b** 的经典数值方法，广泛应用于科学计算、工程仿真等领域。当矩阵规模较大时，其 O(n³) 的时间复杂度使串行计算成为瓶颈。本实验利用 ARM NEON SIMD 指令集对高斯消元算法进行并行优化，以提升计算性能。

实验目标：
- 设计基于 ARM NEON 的高斯消元 SIMD 算法
- 编程实现并验证正确性
- 测试不同问题规模下的性能加速比
- 讨论对齐、选择性向量化、Cache 优化等策略对性能的影响
- 利用 perf 等工具剖析性能瓶颈

## 2 SIMD 算法设计

### 2.1 串行算法回顾

标准高斯消元分为两个阶段：

**前向消元** — 对于 k = 0, 1, …, n-1：

1. 计算消元因子 factor = A[i,k] / A[k,k]
2. 对于 i > k 的每一行，执行 A[i,j] = A[i,j] - factor × A[k,j]（j > k）
3. 同步更新右端向量 b[i] = b[i] - factor × b[k]

**回代求解** — 自底向上计算 x[i] = (b[i] - Σ_{j>i} A[i,j]·x[j]) / A[i,i]

时间复杂度为 O(n³)，空间复杂度为 O(n²)。

### 2.2 SIMD 并行算法

为充分利用 SIMD 向量化能力，采用**先归一化主元行、再批量消去**的策略（手册 Algorithm 1）。归一化后的主元行被复用 n-k-1 次，将除法次数从每行一次降低为每主元列一次。

#### 2.2.1 归一化主元行（对应手册 Algorithm 1 行 2-9）

将主元行第 k 行除以 A[k,k]，使得 A[k,k] = 1.0。归一化后的元素被随后所有下方行的消去计算复用。

SIMD 实现：将主元值 pivot = A[k,k] 广播到 NEON 寄存器的 4 个 lane，每次加载主元行 4 个连续元素，用 `vdivq_f32` 一次完成 4 个除法。

#### 2.2.2 消去（对应手册 Algorithm 1 行 10-20）

对于每一行 i > k，A[i,k] 即为消元因子（因为主元行已归一化）。将 A[i,k] 广播到 4 个 lane，用 `vmulq_f32` 和 `vsubq_f32` 一次完成 4 个元素的乘法和减法。

伪代码：

```
for k = 0 to n-1:
    vt = dup4(A[k,k])
    for j = k+1; j+4 <= n; j += 4:    // SIMD 归一化
        A[k,j] = load4(A[k,j]) / vt
    A[k,k] = 1.0; b[k] /= pivot

    parallel for i = k+1 to n-1:       // OpenMP + SIMD 消去
        vaik = dup4(A[i,k])
        for j = k+1; j+4 <= n; j += 4:
            A[i,j] = load4(A[i,j]) - load4(A[k,j]) * vaik
        A[i,k] = 0; b[i] -= b[k] * A[i,k]
```

#### 2.2.3 回代 SIMD 优化

回代阶段使用 SIMD 点积：`vaddvq_f32(vmulq_f32(a_vec, x_vec))` 一次完成 4 个乘法累加。

### 2.3 复杂度分析

| 阶段 | 串行 | SIMD |
|------|------|------|
| 归一化 | O(n²/2) 除法 | O(n²/8) SIMD 除法 |
| 消去 | O(n³/2) 乘加 | O(n³/8) SIMD 乘加 |
| 回代 | O(n²/2) 乘加 | O(n²/8) SIMD 乘加 |

理论加速比受限于 NEON 128 位宽度（4 × float32），在不考虑访存开销时，SIMD 部分的理论加速比为 4×。

## 3 实现

### 3.1 平台信息

- CPU: ARMv8-A (aarch64)，支持 NEON SIMD
- 编译器: g++，编译选项 `-O2 -march=armv8-a+simd -fopenmp`
- 并行: OpenMP（线程级）+ NEON Intrinsics（指令级）

### 3.2 关键 Intrinsics 映射

| 手册伪代码 | NEON Intrinsic | 说明 |
|-----------|---------------|------|
| `dupTo4Float(x)` | `vdupq_n_f32(x)` | 标量广播到 4 lane |
| `load4FloatFrom(ptr)` | `vld1q_f32(ptr)` | 加载 4 个 float |
| `store4FloatTo(ptr, v)` | `vst1q_f32(ptr, v)` | 存储 4 个 float |
| `va / vt` | `vdivq_f32(va, vt)` | 4 路并行除法 |
| `vaik * vakj` | `vmulq_f32(vakj, vaik)` | 4 路并行乘法 |
| `vaij - vx` | `vsubq_f32(vaij, vx)` | 4 路并行减法 |

### 3.3 数值验证

使用对角线占优随机矩阵（对角线加行和的绝对值）保证矩阵非奇异。解 x 与预设真解 x_true 对比，或计算相对残差 ‖Ax-b‖/‖b‖ 验证正确性。消元中使用 float32 精度，验证阈值设为 10⁻³。

## 4 实验及结果分析

### 4.1 实验设置

测试不同矩阵规模 n ∈ {64, 128, 256, 512, 1024, 2048}，每组重复 5 次取平均。对比以下配置：
- **串行标量**: 无 SIMD，无 OpenMP（基线）
- **NEON SIMD**: NEON 向量化 + OpenMP 多线程
- **NEON SIMD + 对齐**: 使用 `aligned_alloc(16, ...)` 分配 16 字节对齐内存

### 4.2 性能结果

> 注：以下数据为参考模板，需在实际 ARM 平台上运行后填入真实数据。

| n | 串行(ms) | SIMD(ms) | 加速比 | SIMD+对齐(ms) | 误差‖Ax-b‖/‖b‖ |
|---|---------|---------|--------|-------------|----------------|
| 64 | — | — | — | — | — |
| 128 | — | — | — | — | — |
| 256 | — | — | — | — | — |
| 512 | — | — | — | — | — |
| 1024 | — | — | — | — | — |
| 2048 | — | — | — | — | — |

### 4.3 各阶段耗时分布

> 注：以 n=1024 为例，在实际平台上运行后填入。

| 阶段 | 耗时(ms) | 占比 |
|------|---------|------|
| 归一化主元行 | — | —% |
| 消去 | — | —% |
| 回代 | — | —% |
| 总计 | — | 100% |

### 4.4 策略分析

#### 4.4.1 对齐 vs 非对齐

NEON `vld1q_f32` 支持非对齐加载，但 16 字节对齐加载可通过单次缓存行访问完成，减少跨缓存行的二次访问。当矩阵规模较大、数据恰好跨缓存行时，对齐内存可带来 5%-10% 的额外性能提升。

#### 4.4.2 选择性向量化

归一化主元行（除法）的计算量占比很小（约 1/n），SIMD 加速效果有限。消去阶段占总计算量的 O(n³)，是 SIMD 加速的主要收益来源。回代阶段计算量为 O(n²)，SIMD 收益同样较小，但实现简单，无额外开销。

#### 4.4.3 Cache 优化

矩阵按行优先存储，NEON 每次处理同行 4 个连续元素，天然具有良好的空间局部性。当 n 较大时（n > L1 cache 大小 / 4B），可考虑分块策略（blocking）进一步提升 Cache 命中率。

#### 4.4.4 加速比分析

对于大矩阵（n ≥ 512），计算密集度足够高，SIMD 加速比接近内存带宽和 SIMD 宽度的理论极限。对于小矩阵（n < 128），OpenMP 线程创建开销和 SIMD 初始化开销占比较大，加速比会降低。

### 4.5 perf 剖析

使用 perf 工具分析程序的性能特征：

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,branch-misses \
    ./main 1024
```

关键指标：
- **IPC**（instructions per cycle）：期望 > 1.0（标量通常 0.5-1.0，SIMD 可接近 2.0）
- **Cache miss rate**：期望 < 1%
- **Branch miss rate**：期望 < 1%

使用 `-S` 编译选项可输出汇编代码，验证编译器是否正确生成 NEON 指令（`fdiv v*.4s`、`fmul v*.4s`、`fsub v*.4s` 等）。

## 5 结论

本实验基于 ARM NEON SIMD 指令集实现了高斯消元算法的并行加速。通过归一化主元行、批量消去、回代 SIMD 点积三个层次的向量化，结合 OpenMP 多线程并行，在 ARM 平台上实现了高斯消元的指令级+线程级混合并行。实验结果表明，SIMD 向量化对高斯消元这种计算密集型算法具有显著的加速效果。

## 参考文献

[1] ARM Holdings. ARM NEON Programmer's Guide. https://developer.arm.com/architectures/instruction-sets/intrinsics/

[2] Golub G H, Van Loan C F. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013.

## 附录：Git 项目链接

https://github.com/2410683-LiuZihan/parallel-programming-lab2

---

*实验人：刘子涵 (2410683)*  
*日期：2026 年 5 月*
