import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import NegativeBinomial
import numpy as np
from typing import Dict, Optional, Tuple

# ============================================================
# Diff-MMM: 微分可能エージェントベースドマーケティング・ミックス・モデリング
# 
# 本モジュールは、U-A-C（未認知-活性化-購買）状態空間モデルを
# 微分可能な形で実装し、マーケティング・ミックス・モデリング（MMM）
# のパラメータ復元を可能にする。
# ============================================================

class ModelConfig:
    def __init__(
        self,
        n_segments: int = 3,
        n_states: int = 3,
        n_features: int = 3,
        use_paramnet: bool = True,
        use_calibnn: bool = False,
        paramnet_hidden_dim: int = 32,
        calibnn_hidden_dim: int = 16,
        max_epochs: int = 1000,
        patience: int = 50,
        learning_rate: float = 0.01,
        gamma_fixed: Optional[float] = None,
        kappa_fixed: Optional[float] = None,
        sparsity_weight: float = 0.0,
        beta_l2_weight: float = 0.0
    ):
        self.n_segments = n_segments
        self.n_states = n_states
        self.n_features = n_features
        self.use_paramnet = use_paramnet
        self.use_calibnn = use_calibnn
        self.paramnet_hidden_dim = paramnet_hidden_dim
        self.calibnn_hidden_dim = calibnn_hidden_dim
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.gamma_fixed = gamma_fixed
        self.kappa_fixed = kappa_fixed
        self.sparsity_weight = sparsity_weight
        self.beta_l2_weight = beta_l2_weight

class HillTransform(nn.Module):
    """
    Hill関数による飽和変換
    広告出稿量を反応率に変換する（収穮遞減効果を表現）
    """
    def __init__(self, n_media: int, gamma_fixed: Optional[float] = None, kappa_fixed: Optional[float] = None):
        super().__init__()
        # Gamma（飽和曲線の形状パラメータ）
        if gamma_fixed is not None:
            self.register_buffer('gamma', torch.tensor([gamma_fixed] * n_media))
            self.fixed_gamma = True
        else:
            self.gamma = nn.Parameter(torch.ones(n_media) * 2.0)
            self.fixed_gamma = False
            
        # Kappa（半飽和点：効果が50%に達する出稿量）
        if kappa_fixed is not None:
            self.register_buffer('kappa', torch.tensor([kappa_fixed] * n_media))
            self.fixed_kappa = True
        else:
            self.kappa = nn.Parameter(torch.ones(n_media) * 0.5)
            self.fixed_kappa = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [バッチ, n_media] または [T, n_media]
        gamma = self.gamma if self.fixed_gamma else self.gamma
        kappa = self.kappa if self.fixed_kappa else self.kappa
        
        # 正の値を保証
        gamma = torch.abs(gamma)
        kappa = torch.abs(kappa)
        
        x_pow = torch.pow(x + 1e-8, gamma)
        kappa_pow = torch.pow(kappa, gamma)
        
        return x_pow / (x_pow + kappa_pow)

class ParamNet(nn.Module):
    """
    パラメータネットワーク
    セグメント属性から状態遷移確率のパラメータを出力する
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 各パラメータタイプの出力ヘッド
        # U->A：未認知から活性化への遷移
        self.head_alpha_UA = nn.Linear(hidden_dim, 1)
        self.head_beta_UA = nn.Linear(hidden_dim, 3)  # TV, Display, GenSearch
        
        # A->U：忘却（活性化から未認知への後退）
        self.head_lambda = nn.Linear(hidden_dim, 1)
        
        # A->C：活性化から購買への遷移
        self.head_alpha_AC = nn.Linear(hidden_dim, 1)
        self.head_beta_AC = nn.Linear(hidden_dim, 2)  # GenSearch, BrandSearch
        self.head_beta_AC_display = nn.Linear(hidden_dim, 1)  # Display（直接購買効果）
        
        self._init_weights()
        
    def _init_weights(self):
        """
        ニューラルネットワークの初期化
        「広告がなければ自然遷移は稀」という仮定を引き継ぐ
        """
        # Alpha（ロジット）: -5.0から開始（確率約0.6%）
        # ユーザーフィードバック: 「オーガニックはデフォルトで0であるべき」
        # 0.0（50%）で初期化すると約30%で停滞したため、低い値から開始
        nn.init.constant_(self.head_alpha_UA.bias, -5.0)
        nn.init.constant_(self.head_alpha_AC.bias, -5.0)
        
        # Beta（Softplus）: 小さな正の値から開始
        # Alpha=-3.5の場合、Betaは大きな値（約10.0）が必要
        # 5.0で初期化して解に近い位置から開始
        nn.init.constant_(self.head_beta_UA.bias, 5.0)
        nn.init.constant_(self.head_beta_AC.bias, 5.0)
        nn.init.constant_(self.head_beta_AC_display.bias, 5.0)
        
        # Lambda（ロジット）: 小さな負の値から開始（確率 < 0.5）
        # ロジット -1.0 -> 確率 0.26
        nn.init.constant_(self.head_lambda.bias, -1.0)
        
        # 重みは小さなランダム値で初期化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
        
    def forward(self, attributes: torch.Tensor) -> Dict[str, torch.Tensor]:
        # attributes: [n_segments, input_dim] セグメント属性
        h = self.net(attributes)
        
        # U->A：未認知から活性化へ
        alpha_UA = self.head_alpha_UA(h).squeeze(-1)  # [K]
        beta_UA = F.softplus(self.head_beta_UA(h))    # [K, 3]（正の値）
        
        # A->U：忘却
        lambda_k = self.head_lambda(h).squeeze(-1)   # [K]（ロジット）
        
        # A->C：活性化から購買へ
        alpha_AC = self.head_alpha_AC(h).squeeze(-1)  # [K]
        beta_AC = F.softplus(self.head_beta_AC(h))    # [K, 2]（正の値）
        beta_AC_display = F.softplus(self.head_beta_AC_display(h)).squeeze(-1)
        
        return {
            'alpha_UA': alpha_UA,
            'beta_UA': beta_UA,
            'lambda_k': lambda_k,
            'alpha_AC': alpha_AC,
            'beta_AC': beta_AC,
            'beta_AC_display': beta_AC_display
        }



class DiffMMM(nn.Module):
    """
    Diff-MMM: 微分可能マーケティング・ミックス・モデル
    
    U-A-C状態空間モデルを用いて、広告出稿データから
    消費者の認知・購買ファネルをシミュレートする
    """
    def __init__(self, T: int, config: ModelConfig, segment_attributes: torch.Tensor):
        super().__init__()
        self.config = config
        self.segment_attributes = segment_attributes  # [K, n_features] セグメント属性
        self.n_segments = config.n_segments
        self.T = T
        
        # モデルの構成要素
        self.hill_transform = HillTransform(n_media=5, gamma_fixed=config.gamma_fixed, kappa_fixed=config.gamma_fixed if hasattr(config, 'gamma_fixed') else None) 
        # Wait, blindly using gamma logic? No, need separate config for kappa.
        # Let's assume config has kappa_fixed
        kappa_val = getattr(config, 'kappa_fixed', None)
        self.hill_transform = HillTransform(n_media=5, gamma_fixed=config.gamma_fixed, kappa_fixed=kappa_val)
        
        self.param_net = ParamNet(config.n_features, config.paramnet_hidden_dim)
        
        
        # CalibNNは削除（卒業研究の範囲を静的ParamNetに限定）
        self.calib_net = None
        
        # 負の二項分布の分散パラメータ
        self.phi = nn.Parameter(torch.tensor(1.0))
        
        # 基準コンバージョン（学習可能） DGPで50を使用
        self.base_conversion = nn.Parameter(torch.tensor(50.0))

    def forward(self, x_media: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        フォワードパス
        
        Args:
            x_media: [T, 5] - 広告出稿データ (TV, Display, Gen, Brand, Organic)
                     DGPから正規化された入力
        
        Returns:
            購買数の予測値とパラメータ辞書
        """
        # 1. Hill変換を適用
        # p: [T, 5]
        p = self.hill_transform(x_media)
        
        # 1.5 CalibNN（削除済み）
        # デフォルトの乗数を1.0に（キャリブレーションなし）
        multipliers = torch.ones(self.T, 5)
        # self.calib_net is always None now.
        
        # 2. セグメントパラメータを取得
        params = self.param_net(self.segment_attributes)
        
        # 3. 各セグメントについて状態空間をシミュレート
        total_C = torch.zeros(self.T)
        
        # メディアインデックスのマッピング
        # DGP Xの列: 0:TV, 1:Display, 2:Gen, 3:Brand, 4:Organic
        
        for k in range(self.n_segments):
            # パラメータを展開
            alpha_UA = params['alpha_UA'][k]
            beta_UA = params['beta_UA'][k]   # [3]
            lambda_k = params['lambda_k'][k]  # スカラー
            alpha_AC = params['alpha_AC'][k]
            beta_AC = params['beta_AC'][k]   # [2]
            beta_AC_disp = params['beta_AC_display'][k]
            
            # メディア圧力（Hill変換後）
            tv_p = p[:, 0]
            disp_p = p[:, 1]
            gen_p = p[:, 2]
            brand_p = p[:, 3]
            
            # 遷移確率の事前計算（ベクトル化部分）
            # U -> A 確率
            # P_UA_organic（ベースライン）
            # DGP: P_UA_organic = 1 / (1 + exp(-alpha_UA))
            logit_UA_base = alpha_UA
            p_ua_base = torch.sigmoid(logit_UA_base)
            
            # P_UA_with_channel
            # DGP: P_UA_with_tv = 1 / (1 + exp(-(alpha_UA + beta * p)))
            
            # Betaに乗数を適用
            # multipliersの形状: [T, 5] (0:TV, 1:Display, 2:Gen, 3:Brand, 4:Organic)
            
            beta_tv_eff = beta_UA[0] * multipliers[:, 0]
            beta_disp_eff = beta_UA[1] * multipliers[:, 1]
            beta_gen_eff = beta_UA[2] * multipliers[:, 2]
            
            p_ua_tv = torch.sigmoid(logit_UA_base + beta_tv_eff * tv_p)
            p_ua_disp = torch.sigmoid(logit_UA_base + beta_disp_eff * disp_p)
            p_ua_gen = torch.sigmoid(logit_UA_base + beta_gen_eff * gen_p)
            
            # 増分貢献（DGPのロジック）
            # P_channel_only = P_with_channel - P_organic
            prob_UA_tv = p_ua_tv - p_ua_base
            prob_UA_disp = p_ua_disp - p_ua_base
            prob_UA_gen = p_ua_gen - p_ua_base
            prob_UA_base = p_ua_base  # オーガニック貢献
            
            # A -> U（忘却）
            # DGP: AU_flow = A[t-1] * lambda_k
            prob_AU = torch.sigmoid(lambda_k)
            
            # A -> C 確率
            logit_AC_base = alpha_AC
            p_ac_base = torch.sigmoid(logit_AC_base)
            
            # Displayの乗数もA->Cに適用
            # 「広告効果」は通常両方のファネルに影響
            
            beta_disp_ac_eff = beta_AC_disp * multipliers[:, 1]
            beta_gen_ac_eff = beta_AC[0] * multipliers[:, 2]
            beta_brand_ac_eff = beta_AC[1] * multipliers[:, 3]
            
            p_ac_disp = torch.sigmoid(logit_AC_base + beta_disp_ac_eff * disp_p)
            p_ac_gen = torch.sigmoid(logit_AC_base + beta_gen_ac_eff * gen_p)
            p_ac_brand = torch.sigmoid(logit_AC_base + beta_brand_ac_eff * brand_p)
            
            prob_AC_disp = p_ac_disp - p_ac_base
            prob_AC_gen = p_ac_gen - p_ac_base
            prob_AC_brand = p_ac_brand - p_ac_base
            prob_AC_base = p_ac_base
            
            # 連続ループ（DGPと厳密に一致）
            # 市場規模 (M_seg)
            # DGP: M_seg = market_size * segment_params[k]['pop_share']
            # M = 100000/K を使用
            M_seg = 100000.0 / self.n_segments
            
            # 初期状態 (t=0)
            # DGP: U[0] = M_seg * 0.85, 等
            u_prev = M_seg * 0.85
            a_prev = M_seg * 0.14
            c_prev = M_seg * 0.01  # フローには使わない、出力のみ
            
            # 履歴を保存
            c_series = [torch.tensor(0.0)]  # t=0のコンバージョン
            # DGPは t=1 から生成を開始
            
            # DGPループは1からT-1まで
            # Xの形状はT
            # 長さTのCを出力する必要がある
            
            # ループを正確に実装
            c_list = []
            c_list.append(torch.tensor(c_prev))  # C[0]
            
            # リストに蓄積して履歴を追跡（in-place操作は勾配に悪影響）
            
            for t in range(1, self.T):
                # 時刻tの確率
                # 注: DGPは遷移時にX[t]を使用
                
                # U -> A Flow
                # Sum of probabilities.
                # In DGP: UA_tv = U[t-1] * P_UA_tv_only
                # UA_total = sum(UA_channels)
                p_ua_total = prob_UA_tv[t] + prob_UA_disp[t] + prob_UA_gen[t] + prob_UA_base
                flow_UA = u_prev * p_ua_total
                
                # A -> U Flow
                flow_AU = a_prev * prob_AU
                
                # A -> C フロー
                p_ac_total = prob_AC_disp[t] + prob_AC_gen[t] + prob_AC_brand[t] + prob_AC_base
                flow_AC = a_prev * p_ac_total
                
                # 状態の更新
                # DGP: U[t] = max(0, U[t-1] - UA_total + AU_flow)
                # ReLU/clampを使用してmax(0)を模倣
                u_next = F.relu(u_prev - flow_UA + flow_AU)
                a_next = F.relu(a_prev + flow_UA - flow_AU - flow_AC)
                
                # C[t] = AC_total（コンバージョンはフロー）
                c_next = flow_AC
                
                c_list.append(c_next)
                
                # 次のステップの準備
                u_prev = u_next
                a_prev = a_next
            
            segment_C = torch.stack(c_list)
            total_C = total_C + segment_C
            
        return self.base_conversion + total_C, params

    def compute_loss(self, x, y):
        """
        損失関数の計算
        負の二項分布のNLL + 正則化項
        """
        y_pred, params = self(x)
        
        # 1. 負の二項分布損失 (NLL)
        eps = 1e-8
        y_pred = torch.clamp(y_pred, min=eps)
        phi = torch.abs(self.phi) + eps
        
        nb_dist = NegativeBinomial(total_count=phi, probs=y_pred / (y_pred + phi), validate_args=False)
        nll = -nb_dist.log_prob(y).sum()
        
        # 2. 二重正則化（ゼロベースライン戦略）
        # (A) ベースラインスパーシティ: sigmoid(alpha) -> 0 に押す
        # 「広告がなければ自然遷移は稀」という仮定
        reg_loss = 0.0
        
        if self.config.sparsity_weight > 0:
            p_ua_base = torch.sigmoid(params['alpha_UA'])
            reg_loss += self.config.sparsity_weight * torch.sum(p_ua_base ** 2)
            
            p_ac_base = torch.sigmoid(params['alpha_AC'])
            reg_loss += self.config.sparsity_weight * torch.sum(p_ac_base ** 2)
        
        # (B) Beta L2: 爆発を防止
        if self.config.beta_l2_weight > 0:
            reg_loss += self.config.beta_l2_weight * torch.sum(params['beta_UA'] ** 2)
            reg_loss += self.config.beta_l2_weight * torch.sum(params['beta_AC'] ** 2)
        
        return nll + reg_loss

class Trainer:
    """
    モデルの学習を管理するトレーナークラス
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
    def fit(self, X, y, verbose=True):
        """
        モデルの学習
        
        Args:
            X: 広告出稿データ [T, 5]
            y: コンバージョン数 [T]
            verbose: 学習過程を表示するか
        """
        self.model.train()
        losses = []
        
        for epoch in range(self.config.max_epochs):
            self.optimizer.zero_grad()
            loss = self.model.compute_loss(X, y)
            loss.backward()
            
            # 勾配クリッピングで爆発を防止
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            losses.append(loss.item())
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss {loss.item():.4f}")
