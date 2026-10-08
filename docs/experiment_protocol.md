# 实验协议与边界

## 固定条件

原模型是 305,289 参数的条件 U-Net：Flow 训练 15,000 次更新，DDPM 20,000 次更新。
二者训练预算不同；不能用该结果主张训练效率优劣。固定 55,000/5,000 训练/验证划分，
官方测试集不用于训练或选生成器 checkpoint。Flow 和 DDPM 的 MSE 预测目标不同。

历史续训选择使用 1,024 个验证图片 × 两组固定噪声/时间，按验证 MSE 选择最佳 EMA。
早期使用 256 个图片的一组固定验证噪声；展示时分开画曲线。
新版可选训练是独立从头训练、随机打乱遍历，不能把其 step/epoch 与历史采样协议混为一谈。

## 正式 ODE 误差实验

固定冻结 Flow 权重、随机种子与 100 个独立高斯初值（每类 10 个）。
Euler / Heun / RK4 比较 N=5..640；参考 RK4 1,280/2,560/5,120 步。
误差在未裁剪原始状态上计算。观测阶只在超过参考分辨率、且进入足够小步长区间时拟合。
细化差异不能单独提供严格参考误差界。线性测试中的刚性不是网络刚性的证据。

等 NFE 比较 20/40/80/160/320，热身、同步设备，随机化计时顺序，保留每次计时。

从项目根目录运行 Python 批量入口，明确指定新的输出目录：

```bash
python scripts/run_numerics.py --out outputs/numerical_analysis/new_run
```

默认完整实验较耗时；小规模入口用于检查管线，不用于替代正式结论：

```bash
python scripts/run_numerics.py --pilot --out outputs/numerical_analysis/pilot_new
```

已有缓存与配置不兼容时脚本会拒绝使用。相同 GPU 种子不保证跨平台逐位一致。
`results/numerical_analysis/` 是原运行输出的精选导出，不是新脚本自动重跑的结果。
完整分析脚本目前自动选择 MPS 或 CPU；交互面板支持显式 CPU/MPS/CUDA 模型。

## 生成质量

`scripts/mnist_quality.py` 使用冻结分类器倒数第二层的 64D 特征，计算按类的 RBF MMD²、
特征 precision/recall 等。该 MMD 不是标准 Inception FID/KID；无偏估计可能略为负值。
5 个独立噪声块的标准差不是训练种子稳健性，也不是置信区间。
分类器标签遵从性不能替代完整生成质量，更不能提供生物学解释。

完整质量评估 18 组 × 5,000 个生成样本 = 90,000 张图片。先 pilot，再按需要运行 final：

```bash
python scripts/mnist_quality.py --stage pilot
python scripts/mnist_quality.py --stage final
```
