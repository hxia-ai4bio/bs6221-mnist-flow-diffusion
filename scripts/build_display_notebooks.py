"""Build lightweight notebooks that call Python APIs; no ML implementation in cells."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]


def build():
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    setup=code('''from pathlib import Path
import os, sys, torch
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'bs6221_flow').is_dir())
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
torch.set_num_threads(4)
from bs6221_flow.sampling import load_flow, generate_images, compare_solvers, solve_ode
from bs6221_flow.training import train_model
from bs6221_flow.presentation import show_training, forward_demo, probability_paths, generation_demo, diffusion_demo, show_saved_numerics
from bs6221_flow.widgets import generation_panel, numerics_panel
from IPython.display import display
DEVICE = os.environ.get('BS6221_DEVICE', 'auto')
FLOW_WEIGHTS = None  # None: frozen 15,000-step EMA; or your training run's best.pt
model = load_flow(FLOW_WEIGHTS, DEVICE)
print('Device:', next(model.parameters()).device, '| weights SHA-256:', model.flow_weight_sha256)''')
    training=[md('''## 环境与可选训练
先按 README 创建环境。默认直接加载预训练模型，下面不会自动训练。
历史训练曲线来自保存的 checkpoint；Flow 的速度 MSE 与 DDPM 的噪声 MSE 不可直接比较。
历史训练有放回抽样，图中 epoch 是等效遍历次数；新训练按随机打乱后的完整数据遍历。
续训的验证协议与早期不同，分开画线；没有记录的训练损失保持缺失。'''),
              code("show_training(kind='flow')\nshow_training(kind='diffusion')"),
              code('''RUN_TRAINING = os.environ.get('BS6221_RUN_TRAINING', '0') == '1'
TRAIN_CONFIG = dict(kind='flow', steps=1000, batch_size=64, learning_rate=3e-4,
                    ema_decay=.995, seed=42, validation_every=250,
                    validation_samples=256, device_preference=DEVICE)
# To train full epochs: remove steps and add epochs=1. Each epoch covers 55,000 images.
print('Optional training parameters:', TRAIN_CONFIG)
if RUN_TRAINING:
    training_folder, history = train_model(**TRAIN_CONFIG)
    show_training(history=history, kind=TRAIN_CONFIG['kind'])
    if TRAIN_CONFIG['kind'] == 'flow':
        model = load_flow(training_folder / 'best.pt', DEVICE)''')]
    paths=[md('''## 1. 图片 → 噪声 → 概率路径 → 模型生成
先看真实训练图片逐渐加噪。Flow 插值使用固定噪声；DDPM 链每步引入新噪声。
图片到噪声记 s∈[0,1]；Flow 训练/生成记 t=1−s，噪声 t=0，图片 t=1。'''),
           code('forward_demo(digit=3, seed=42)'),
           md('''### 样本空间中的构造概率路径
每张图片是 784 维向量。这里只用训练图片拟合固定 PCA 轴，投影噪声到图片的经验样本路径。
颜色表示配对的真实数字类别；这不是网络学到的生成轨迹，也不是 784 维概率密度。'''),
           code('probability_paths(seed=42, samples=300)'),
           md('''### 模型实际怎样生成图片
Flow 每一步用网络预测速度，再由求解器更新状态。DDPM 预测噪声并执行反向随机采样。
下面使用新噪声，不使用前面真实图片的终点。DDPM 的“采样进度”不是 Flow 的时间参数。'''),
           code('generation_demo(net=model, digit=3, seed=42, steps=40)\ndiffusion_demo(digit=3, seed=42, step_seed=43, device_preference=DEVICE)')]
    interactive=[md('''## 2. 选择数字和噪声，再看生成过程
选择数字、噪声种子、求解器和步数。先预览噪声，再生成图片；相同模型、标签和初始噪声可重复。
相同 NFE 比较会使用同一噪声；改变噪声尺度属于扰动实验，默认标准高斯尺度为 1。
实际路径来自积分过程，不是图片插值。可调用 generate_images(..., noise=你的张量, net=model)。'''),
                 code('playground = generation_panel(model)\ndisplay(playground)')]
    numerical=[md('''## 3. ODE 数值分析：参数、成本和收敛性
NFE = 步数 × 每步网络调用数：Euler 1、Heun 2、RK4 4。相同步数与相同 NFE 是不同预算。
下面先展示原冻结模型的实验记录；它们不是这次重新计算的结果，也不对应自训练模型。
理论阶需要正则性和渐近步长条件。RMSE 相对于近似参考解，不是真实图片误差。
稳定性测试中的线性刚性不证明这个神经网络 ODE 刚性。'''),
               code('show_saved_numerics()'),
               md('''### 自定义小规模数值实验（点击才计算）
改变样本数、步数列表、NFE、参考分辨率与计时重复。各方法使用同一个当前模型、标签和噪声。
CPU float64 RK4 两级加密检验参考分辨率；未通过时不要报告可信收敛阶。
小样本适合探索，正式结论应使用完整独立样本与更精细参考。'''),
               code('ode_panel = numerics_panel(model)\ndisplay(ode_panel)')]
    variants={
      '00_project_demo.ipynb': [md('# MNIST Flow / Diffusion：可选训练、生成与 ODE 数值分析\n\n实现全部在 `bs6221_flow/*.py`；本 Notebook 只设置参数、调用和展示。'),setup,*training,*paths,*interactive,*numerical],
      '01_probability_space_flow_diffusion.ipynb':[md('# 从图片到噪声与概率路径'),setup,*paths],
      '02_conditional_flow_diffusion.ipynb':[md('# 条件训练与数字生成'),setup,*training,*interactive],
      '03_flow_numerical_analysis.ipynb':[md('# Flow ODE 数值分析'),setup,*numerical],
    }
    for name,cells in variants.items():
        nb=nbf.v4.new_notebook(cells=cells)
        nb.metadata['kernelspec']={'display_name':'BS6221 Flow (shared)','language':'python','name':'bs6221-flow-shared'}
        nb.metadata['language_info']={'name':'python','version':'3.12'}
        nbf.write(nb,ROOT/'notebooks'/name)
    # Full repo is the sharing unit; this old entry stays usable with shared Python code.
    nb=nbf.v4.new_notebook(cells=[md('# 小组交互入口\n\n请保留整个仓库文件夹；完整教学入口为 notebooks/00_project_demo.ipynb。'),setup,*interactive,*numerical])
    nb.metadata=variants and nbf.read(ROOT/'notebooks/00_project_demo.ipynb',as_version=4).metadata
    nbf.write(nb,ROOT/'Flow_Group_Package/Flow_Group.ipynb')
    print('Built five lightweight display notebooks.')

if __name__=='__main__':build()
