# MNIST Flow / Diffusion：Python 与 Bash 复现

用小型条件 U-Net 展示图片加噪、样本空间的概率路径、噪声生成数字，以及 Euler / Heun / RK4 的 ODE 数值分析。
全部通过 Python 和 Bash 运行，图表保存为 PNG，数据保存为 CSV，参数与来源保存为 JSON。
默认加载预训练权重；训练和新数值实验均需显式运行。

## 1. 配置环境与下载权重

```bash
git clone https://github.com/hxia-ai4bio/bs6221-mnist-flow-diffusion.git
cd bs6221-mnist-flow-diffusion
export PYTHONNOUSERSITE=1
conda env create -f environment.yml
conda activate bs6221-flow
bash scripts/setup.sh
```

已经有 `bs6221-flow` 环境时，激活后执行 `git pull` 和 `bash scripts/setup.sh`。
安装前禁用用户全局包，避免全局目录里的依赖被误认为已经装入 Conda 环境。
脚本不会卸载旧环境中的包；新环境只安装 Python 运行所需依赖。

无 Conda 时，使用 Python 3.10–3.12：

```bash
python -m venv .venv
source .venv/bin/activate
bash scripts/setup.sh
```

Bash 脚本适用于 macOS、Linux 或安装了 Bash 的系统。Windows 也可逐条运行下面的 Python 命令。
NVIDIA 用户先按 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) 安装与驱动匹配的 PyTorch。
CPU 已验证；Windows/Linux/CUDA 未实机验证，跨平台不保证逐位相同或耗时相同。

模型公开托管于 [Hugging Face](https://huggingface.co/hx03-info/bs6221-mnist-flow-diffusion)。
`models/hub.json` 固定提交 `1edc45c59e32eac9c55ddad3485eb4949a20a5d4`，下载后验证 SHA-256。
MNIST 官方压缩数据及 seed-42 划分随仓库提供。也可单独执行：

```bash
python scripts/download_weights.py
python scripts/restore_shared_data.py
```

下载脚本不覆盖已有但校验不匹配的权重。安装依赖和首次下载权重需要联网。

## 2. 可选：训练模型与查看 epoch / error

已有权重可运行所有演示。先用小规模训练检查流程：

```bash
python scripts/train.py --model flow --steps 1000 --batch-size 64 --lr 0.0003 --device auto
# 或按完整 epoch 训练；--epochs 与 --steps 二选一
python scripts/train.py --model flow --epochs 1 --batch-size 64
python scripts/train.py --model diffusion --steps 1000
```

终端打印 step、epoch、训练 MSE 和验证 MSE。每个 epoch 随机打乱并遍历 55,000 张训练图片。
验证只使用固定 5,000 张验证图片，不读取官方测试集；验证噪声/时间固定。
Flow 预测速度、DDPM 预测噪声，它们的 MSE 不能直接比较。短训练不能证明生成质量。
每次训练保存到独立 `outputs/training/<模型>_<运行编号>/`，不覆盖非空目录：

| 文件 | 内容 |
| --- | --- |
| `config.json` | seed、batch、学习率、EMA、数据划分哈希等 |
| `history.csv` | step、epoch、训练/验证 MSE、梯度范数及耗时 |
| `summary.json` | 运行结果 |
| `best.pt` | 按验证 MSE 选择的 EMA 推理权重 |
| `latest.pt` | 最后模型、EMA、优化器和随机状态；尚未实现续训命令 |

训练结束自动保存曲线。另行查看原训练记录或自己的训练目录：

```bash
python scripts/demo.py training-curves --out outputs/curves
python scripts/demo.py training-curves --history outputs/training/你的运行目录 --out outputs/my_curves
```

原训练有放回抽样，epoch 为等效遍历次数；早期与续训验证协议不同，分开画线。
续训未记录的训练损失保持缺失；原 Flow 15,000 与 DDPM 20,000 updates 不是等训练预算比较。

## 3. 图片到噪声、概率路径、模型生成

一次运行全部教学展示（不训练、不重算数值参考解）：

```bash
bash scripts/run_demo.sh all --digit 3 --seed 42 --out outputs/demo
# 等价 Python 命令
python scripts/demo.py all --digit 3 --seed 42 --out outputs/demo
```

运行结束后打开 `outputs/demo/` 查看图片；macOS 可执行 `open outputs/demo`。

也可逐项运行：

```bash
python scripts/demo.py forward --digit 3 --seed 42 --out outputs/forward
python scripts/demo.py paths --path-samples 300 --seed 42 --out outputs/paths
python scripts/demo.py generate --digit 3 --seed 42 --method Heun --steps 40 --out outputs/generation
python scripts/demo.py diffusion --digit 3 --seed 42 --step-seed 43 --out outputs/ddpm
```

`forward` 展示真实训练图片逐渐加噪：Flow 使用固定噪声插值，DDPM 链每步引入新噪声。
`paths` 用训练图片拟合固定 PCA 轴，投影 784 维样本的噪声到图片路径。
这是构造的经验概率路径及二维投影，不是完整概率密度或网络实际生成轨迹。
图片到噪声的进度记 s；Flow 生成的时间 t=1−s，噪声 t=0，图片 t=1。
`generate` 展示实际 Flow 积分状态；`diffusion` 展示反向随机采样进度，它不是 Flow 时间。

## 4. 选择数字和噪声

数字由 `--digit` 选择，初始噪声由 `--seed` 固定；`--count` 控制图片数。

```bash
bash scripts/run_demo.sh generate --digit 7 --seed 123 --count 4 --method RK4 --steps 40 --out outputs/digit7
```

保存初始噪声预览、最终图片、真实生成轨迹、`initial_noise.npy`、`generation_states.pt` 和 `generation_params.json`。
默认标准高斯噪声尺度为 1；`--noise-scale` 改变尺度，用于扰动实验。
复用同一份噪声，改变数字或求解器：

```bash
python scripts/demo.py generate --digit 2 --count 4 --noise-file outputs/digit7/initial_noise.npy --method Heun --steps 40 --out outputs/same_noise_digit2
python scripts/demo.py compare --digit 7 --seed 123 --count 4 --nfe 40 --out outputs/compare
```

自定义 `.npy` 噪声形状为 `[count,1,28,28]`，只用于 `generate`；提供文件时保持 `--noise-scale 1`。
`compare --nfe 40` 在同一模型、标签和噪声下使用等 NFE，预算须为 4 的倍数。
省略 `--nfe` 时按 `--steps` 比较相同步数。图像差异本身不是数值误差估计。
使用自己训练的模型：

```bash
python scripts/demo.py generate --weights outputs/training/你的Flow运行目录/best.pt --digit 3 --seed 42
python scripts/demo.py diffusion --diffusion-weights outputs/training/你的DDPM运行目录/best.pt --digit 3
```

Flow 积分状态不裁剪；显示灰度范围固定为 [-1,1]。固定参数可重复，但跨设备浮点结果可能不同。

## 5. ODE 参数、成本与收敛性

先查看原冻结模型的已保存实验：

```bash
python scripts/demo.py saved --out outputs/original_results
```

![同一初始噪声的求解器生成](results/numerical_analysis/07_same_noise_image_grid.png)

显式运行新的小规模探索：

```bash
python scripts/demo.py numerics --samples 2 --digit 3 --seed 42 \
  --step-list 4,8,16,32 --nfe-list 4,8,20,40 \
  --reference-steps 128,256 --repeats 3 --device cpu --out outputs/numerics
```

保存步长–RMSE、NFE–RMSE、NFE–耗时图，`convergence.csv`、`same_nfe.csv`、`orders.csv` 和 `reference_report.json`。
Euler 每步调用网络 1 次，Heun 2 次，RK4 4 次：NFE = 步数 × 阶段数。
各方法使用同一标签和初始噪声；计时预热后重复，排除模型加载、画图和参考解计算。
CPU float64 RK4 两级加密用于参考分辨率诊断。默认参考不保证通过，未通过时不报告观测阶，
需增大参考分辨率；即使通过，也要检查渐近步长区间。
RMSE 相对于同一网络 ODE 的近似参考解，不是真实图片误差；加密差异不是严格误差界。
小样本只用于探索；原正式实验使用 100 个独立噪声（每类 10 个）。

![误差与计算成本](results/numerical_analysis/10_accuracy_cost_tradeoff.png)

完整 ODE 重算见 [实验协议](docs/experiment_protocol.md)。完整生成质量评价成本较高，应先验证小规模流程。
`results/` 是原实验记录，有来源信息；本次工程验证不等于重新训练或重跑正式实验。

## 6. 目录与记录

```text
bs6221_flow/              # 数据、模型、训练、采样、数值分析和 PNG 输出
scripts/demo.py          # 所有教学展示的命令入口
scripts/train.py         # 可选训练
scripts/setup.sh         # 依赖、权重下载、数据恢复
scripts/run_demo.sh      # Bash 演示入口
models/hub.json          # Hugging Face 固定版本
checkpoints/frozen/      # 下载后的权重、冻结定义与校验清单
Flow_Group_Package/data/ # 压缩 MNIST 与固定划分
results/                 # 原训练和数值实验记录
docs/                    # 实验协议、模型说明
outputs/                 # 本次运行结果；不上传 GitHub
```

运行覆盖同名输出文件时请使用新的 `--out` 目录，保留比较实验。
历史 `finetune_conditional.py` 等脚本需要本地完整训练 checkpoint；远程仅提供 EMA 推理权重。
原独立分享包的验证记录仍保留，当前工作流验证以 `reproduction_check.json` 为准。

## 参考

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [MNIST](https://yann.lecun.org/exdb/mnist/)
