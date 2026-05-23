## 环境要求

- Python 3.9（3.8+ 一般可用）
- 见 `requirements.txt`：**必需**仅为 `torch`、`numpy`、`scikit-learn`（及 sklearn 常见依赖 `scipy` 等）
- GPU 可选：`P0_config.py` 中 `--gpu 0` 使用 CUDA；无 GPU 时设 `--gpu -1` 使用 CPU

```bash
pip install -r requirements.txt
```

有 NVIDIA GPU 时，请按 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装带 CUDA 的 `torch`，再安装其余必需包。

## 目录结构

```
.
├── P0_config.py          # 命令行参数与设备选择
├── P1_dataUtil.py        # 数据路径、技能/题目数量、CSV 读取
├── P2_dataset.py         # Dataset 与 batch padding
├── P3_main.py            # 训练入口（python P3_main.py）
├── P4_model_inSkillQues.py   # 双流编码器 + 预测头 + 反事实生成
├── P5_trainUtil.py       # 训练循环、早停、指标打印
├── P6_utils.py           # AUC/Acc、损失与对比损失
├── requirements.txt
├── README.md
└── dataset/              # 预处理后的 CSV（按数据集分子目录）
    └── assist2012/       # 已附带，便于快速跑通
        ├── train1.csv
        ├── eval1.csv
        └── test1.csv
```

## 快速开始（Assist12）
仓库已包含 **Assist2012** 划分数据，无需额外下载即可快速验证：

```bash
python P3_main.py --dataset assist2012 --gpu 0
```

## 常用运行示例
```bash
# Assist12
python P3_main.py --dataset assist2012 --gpu 0 --bsz 16 --lr 1e-4

# CPU
python P3_main.py --dataset assist2012 --gpu -1
```

### 主要参数（`P0_config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `assist0910` | 数据集名称，见下表 |
| `--min_seq_len` | `10` | 最短序列长度，低于此长度的学生样本丢弃 |
| `--n_epoch` | `300` | 最大训练轮数 |
| `--bsz` | `16` | batch size |
| `--lr` | `1e-4` | 学习率 |
| `--gpu` | `0` | GPU 编号；`-1` 为 CPU |

## 其他数据集下载
| 数据集 | 链接 |
|--------|------|
| Assist09 | https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data |
| Assist15 | https://sites.google.com/site/assistmentsdata/home/2015-assistments-skill-builder-data |
| Assist17 | https://sites.google.com/view/assistmentsdatamining |
| Algebra05 | https://pslcdatashop.web.cmu.edu/KDDCup |

## 训练日志说明

每个 epoch 会打印：

1. 单独一行 `epoch N`
2. `train` / `eval` / `test` 的 Acc、AUC
3. 若验证集创新高：`✓ 验证集新最佳 (...)` 
4. 若未创新高：`早停监控: k/10`（满 10 轮触发早停）

训练结束后输出最佳验证集 epoch 及该 epoch 对应的测试集指标。

