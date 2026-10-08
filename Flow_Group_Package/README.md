# 小组演示入口

新版 Notebook 调用仓库根目录的 `bs6221_flow/*.py`，请下载整个 GitHub 仓库。
优先打开 `notebooks/00_project_demo.ipynb`：可选训练、训练曲线、图片到噪声、
概率路径、模型生成、数字与噪声交互、ODE 数值分析。

环境配置、数据和权重下载步骤见根目录 README。模型实现和训练逻辑不再嵌入 Notebook。
本目录的 `data/` 仍提供完整 MNIST 压缩数据和原固定划分；`package_manifest.json`
记录原数据与权重校验。`validation_report.json` 是旧独立版本的历史验证，
新版验证记录见根目录 `reproduction_check.json`。

原 ZIP 是旧版本，保留在本机备份，不作为新版分享入口。
