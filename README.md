# Diff-MMM: Differentiable Agent-Based Marketing Mix Modeling

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 概要 / Overview

**Diff-MMM**は、微分可能エージェントベースモデリング（Differentiable Agent-Based Modeling）を用いた新しいマーケティング・ミックス・モデリング（MMM）フレームワークです。

消費者の認知・購買ファネルを**U-A-C（未認知-活性化-購買）状態空間モデル**として記述し、構造的制約（Structural Masking）の下でパラメータ復元とアトリビューション推定を行います。

### 主な特徴
- 🧠 **状態空間モデル**: 消費者の認知状態の動態を明示的にモデル化
- 🔗 **構造的マスキング**: ドメイン知識に基づく因果パスの制約
- 📊 **微分可能シミュレーション**: 勾配ベースの最適化でパラメータを学習
- 🎯 **スピルオーバー効果**: TV→検索などの間接効果を構造的に捉える

## インストール / Installation

```bash
git clone https://github.com/shota-0712/diff-mmm.git
cd diff-mmm
pip install -r requirements.txt
```

## 使用方法 / Usage

### 1. データ生成（シミュレーション）

```python
from dgp.state_space_dgp import StateSpaceUACDGP

# DGPを作成
dgp = StateSpaceUACDGP(
    T=730,  # 730日間
    market_size=100000,
    n_segments=3
)

# データを生成
X, y, attributions = dgp.generate()
```

### 2. モデルの学習

```python
from src.diff_mmm import DiffMMM, ModelConfig, Trainer
import torch

# 設定
config = ModelConfig(
    n_segments=3,
    n_states=3,
    n_features=3,
    max_epochs=1000,
    learning_rate=0.01
)

# セグメント属性（ダミー）
segment_attributes = torch.eye(3)

# モデルを初期化
model = DiffMMM(T=730, config=config, segment_attributes=segment_attributes)

# 学習
trainer = Trainer(model, config)
trainer.fit(X_tensor, y_tensor)
```

### 3. パラメータ復元の確認

```python
# 推論
with torch.no_grad():
    y_pred, params = model(X_tensor)

# パラメータを確認
print(f"Lambda (忘却率): {torch.sigmoid(params['lambda_k']).mean().item():.3f}")
print(f"Alpha_UA (ベースライン): {torch.sigmoid(params['alpha_UA']).mean().item():.3f}")
```

## ディレクトリ構造 / Directory Structure

```
diff-mmm/
├── src/
│   └── diff_mmm.py      # メインモデル（DiffMMM, ParamNet, HillTransform）
├── dgp/
│   └── state_space_dgp.py  # データ生成プロセス（U-A-C状態空間）
├── experiments/
│   └── experiment_1_parameter_recovery.py  # パラメータ復元実験
├── README.md
├── requirements.txt
└── .gitignore
```

## 理論的背景 / Theoretical Background

本フレームワークは以下の研究に基づいています：

1. **SIRモデルのマーケティング応用** - 疫学のSIR（Susceptible-Infected-Recovered）モデルを消費者ファネルに対応付け
2. **GradABM** - 微分可能エージェントベースモデリング（Chopra et al., AAMAS 2023）
3. **構造的識別性** - ドメイン知識に基づく構造制約による多重共線性の緩和

## 引用 / Citation

```bibtex
@thesis{horie2025diffmmm,
  author = {堀江 祥汰},
  title = {Diff-MMM: 微分可能エージェントベースモデリングによる構造的マーケティング・ミックス・モデリング},
  school = {東京理科大学},
  year = {2025}
}
```

## ライセンス / License

MIT License - 詳細は [LICENSE](LICENSE) を参照してください。

## 著者 / Author

**堀江 祥汰** (Shota Horie)
- 東京理科大学 経営学部 ビジネスエコノミクス学科
- Email: 8722165@ed.tus.ac.jp
