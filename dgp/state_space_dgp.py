#!/usr/bin/env python3
"""
完全版DGP - 日次チャネル別貢献度追跡 & DiffMMM互換正規化
==========================================================

【特徴】
1. 入力データの正規化 (Mean=1.0) -> DiffMMMの入力スケールと整合
2. Hill Transform (Gamma=2.0, Kappa=0.5) をシミュレーションに適用
3. U→A, A→C の遷移をチャネル別に分解して貢献度を記録
"""

import numpy as np
import torch
from typing import Dict, List, Optional


def create_segment_attributes(n_segments: int = 3) -> torch.Tensor:
    """セグメント属性を生成"""
    if n_segments == 3:
        return torch.tensor([
            [0.30, 0.65, 0.85],  # 若年層
            [0.50, 0.50, 0.50],  # 中間層
            [0.70, 0.45, 0.30],  # 高齢層
        ], dtype=torch.float32)
    else:
        return torch.rand(n_segments, 3)


def hill_transform(x: np.ndarray, gamma: float = 2.0, kappa: float = 0.5) -> np.ndarray:
    """
    Hill変換：広告出稿量 → 広告圧力
    p = x^γ / (x^γ + κ^γ)
    """
    x_pow = np.power(x + 1e-8, gamma)
    kappa_pow = np.power(kappa, gamma)
    return x_pow / (x_pow + kappa_pow)


def generate_segment_params(segment_attributes: torch.Tensor) -> List[Dict]:
    """セグメント別パラメータ生成"""
    K = segment_attributes.shape[0]
    segment_params = []
    
    for k in range(K):
        attrs = segment_attributes[k].numpy()
        age, female, urban = attrs[0], attrs[1], attrs[2]
        
        # U → A パラメータ (正規化・Hill変換後の圧力 ~0.8 に対して調整)
        # logit = alpha + beta * 0.8
        # Staratgy: alpha=-4.0, beta ~ 5.0 -> logit=0 -> prob=0.5
        
        # 忘却
        lambda_k = 0.25 - 0.10 * age
        
        # U->A Parameters
        alpha_UA = -3.5 # Organic目標 2-5% (-4.5 -> -3.5)
        
        # Beta (広告効果): ベースラインがない分、強力にする必要がある
        # Hill(x)=1.0 のとき、Sigmoid(-10 + beta) が十分な確率になるように
        # beta=10 -> sigmoid(0) = 0.5
        # beta=12 -> sigmoid(2) = 0.88
        base_beta_tv = 16.0    # U->Aを増やす (13 -> 16)
        base_beta_disp = 10.0  # Maintain
        base_beta_gen = 12.0   # Maintain
        
        # 異質性の反映（係数を調整）
        # 例: 若年層(attr[0]小)はNetに敏感、高齢層はTVに敏感
        # beta_tv: Age(0)に比例（高齢ほど効く）
        beta_tv_UA = base_beta_tv * (0.8 + 0.4 * float(attrs[0])) 
        # beta_disp: Age(0)に反比例（若年ほど効く）
        beta_disp_UA = base_beta_disp * (1.2 - 0.4 * float(attrs[0]))
        # beta_gen: Urban(2)に比例（都市部で効く）
        beta_gen_UA = base_beta_gen * (0.8 + 0.4 * float(attrs[2]))
        
        # A->C Parameters
        alpha_AC = -3.5 # U->Aに合わせて調整
        beta_disp_AC = 0.5 + 0.5 * urban      # 0.5 - 1.0 (補助的)
        beta_gen_AC = 4.0 + 1.0 * urban       # 4.0 - 5.0
        beta_brand_AC = 12.0 + 2.0 * female    # 微調整 (14.0 -> 12.0)
        
        params = {
            # オーガニック（異質性）のベースライン
            'alpha_UA': alpha_UA, 
            'alpha_AC': alpha_AC,
            
            # チャネル効果
            'beta_tv_UA': beta_tv_UA,
            'beta_disp_UA': beta_disp_UA,
            'beta_gen_UA': beta_gen_UA,
            'lambda_k': lambda_k,
            'beta_disp_AC': beta_disp_AC,
            'beta_gen_AC': beta_gen_AC,
            'beta_brand_AC': beta_brand_AC,
            'pop_share': 1.0 / K,
        }
        
        segment_params.append(params)
    
    return segment_params


def _simulate_segment_with_attribution(
    params: Dict,
    X_norm: np.ndarray,
    M_seg: float,
    time_varying_multipliers: Optional[np.ndarray] = None
) -> Dict:
    """
    1セグメントのシミュレーション with 詳細な貢献度追跡
    """
    T = X_norm.shape[0]
    
    # 状態変数
    U = np.zeros(T)
    A = np.zeros(T)
    C = np.zeros(T)
    
    # 初期状態
    U[0] = M_seg * 0.85
    A[0] = M_seg * 0.14
    C[0] = M_seg * 0.01
    
    # チャネル別累積貢献
    contrib_UA_tv = 0.0
    contrib_UA_disp = 0.0
    contrib_UA_gen = 0.0
    contrib_UA_organic = 0.0
    
    contrib_AC_disp = 0.0
    contrib_AC_gen = 0.0
    contrib_AC_brand = 0.0
    contrib_AC_organic = 0.0
    
    for t in range(1, T):
        # 広告圧力（Hill Transform済み）
        tv_p = X_norm[t, 0]
        disp_p = X_norm[t, 1]
        gen_p = X_norm[t, 2]
        brand_p = X_norm[t, 3]
        
        # ========================================
        # U → A 遷移の分解
        # ========================================
        alpha_UA = params['alpha_UA']
        
        # オーガニック（ベースライン）
        P_UA_organic = 1 / (1 + np.exp(-alpha_UA))
        
        # TV寄与
        beta_tv_t = params['beta_tv_UA'] * time_varying_multipliers[t, 0] if time_varying_multipliers is not None else params['beta_tv_UA']
        P_UA_with_tv = 1 / (1 + np.exp(-(alpha_UA + beta_tv_t * tv_p)))
        P_UA_tv_only = P_UA_with_tv - P_UA_organic
        
        # Display寄与
        P_UA_with_disp = 1 / (1 + np.exp(-(alpha_UA + params['beta_disp_UA'] * disp_p)))
        P_UA_disp_only = P_UA_with_disp - P_UA_organic
        
        # GenSearch寄与
        P_UA_with_gen = 1 / (1 + np.exp(-(alpha_UA + params['beta_gen_UA'] * gen_p)))
        P_UA_gen_only = P_UA_with_gen - P_UA_organic
        
        # 実際の遷移人数（チャネル別）
        UA_tv = max(0, U[t-1] * P_UA_tv_only)
        UA_disp = max(0, U[t-1] * P_UA_disp_only)
        UA_gen = max(0, U[t-1] * P_UA_gen_only)
        UA_organic = max(0, U[t-1] * P_UA_organic)
        
        UA_total = UA_tv + UA_disp + UA_gen + UA_organic
        
        # 累積
        contrib_UA_tv += UA_tv
        contrib_UA_disp += UA_disp
        contrib_UA_gen += UA_gen
        contrib_UA_organic += UA_organic
        
        # ========================================
        # 忘却 (A → U)
        # ========================================
        AU_flow = A[t-1] * params['lambda_k']
        
        # ========================================
        # A → C 遷移の分解
        # ========================================
        alpha_AC = params['alpha_AC']
        
        # オーガニック
        P_AC_organic = 1 / (1 + np.exp(-alpha_AC))
        
        # Display寄与
        P_AC_with_disp = 1 / (1 + np.exp(-(alpha_AC + params['beta_disp_AC'] * disp_p)))
        P_AC_disp_only = P_AC_with_disp - P_AC_organic
        
        # GenSearch寄与
        P_AC_with_gen = 1 / (1 + np.exp(-(alpha_AC + params['beta_gen_AC'] * gen_p)))
        P_AC_gen_only = P_AC_with_gen - P_AC_organic
        
        # BrandSearch寄与
        P_AC_with_brand = 1 / (1 + np.exp(-(alpha_AC + params['beta_brand_AC'] * brand_p)))
        P_AC_brand_only = P_AC_with_brand - P_AC_organic
        
        # 実際の遷移人数（チャネル別）
        AC_disp = max(0, A[t-1] * P_AC_disp_only)
        AC_gen = max(0, A[t-1] * P_AC_gen_only)
        AC_brand = max(0, A[t-1] * P_AC_brand_only)
        AC_organic = max(0, A[t-1] * P_AC_organic)
        
        AC_total = AC_disp + AC_gen + AC_brand + AC_organic
        
        # 累積
        contrib_AC_disp += AC_disp
        contrib_AC_gen += AC_gen
        contrib_AC_brand += AC_brand
        contrib_AC_organic += AC_organic
        
        # ========================================
        # 状態更新
        # ========================================
        U[t] = max(0, U[t-1] - UA_total + AU_flow)
        A[t] = max(0, A[t-1] + UA_total - AU_flow - AC_total)
        C[t] = max(0, AC_total)
    
    # 総購買人数
    total_conversions = contrib_AC_disp + contrib_AC_gen + contrib_AC_brand + contrib_AC_organic
    
    # 貢献度（%）
    attribution = {
        'TV_awareness': contrib_UA_tv,
        'Display_awareness': contrib_UA_disp,
        'GenSearch_awareness': contrib_UA_gen,
        'Organic_awareness': contrib_UA_organic,
        'Display_conversion': contrib_AC_disp,
        'GenSearch_conversion': contrib_AC_gen,
        'BrandSearch_conversion': contrib_AC_brand,
        'Organic_conversion': contrib_AC_organic,
        'total_conversions': total_conversions,
    }
    
    return {
        'states': {'U': U, 'A': A, 'C': C},
        'attribution': attribution
    }


def generate_dgp_data(
    n_days: int = 730,
    seed: int = 42,
    segment_attributes: Optional[torch.Tensor] = None,
    market_size: int = 100000,
    base_conversion: float = 50.0,
    noise_scale: float = 0.1,
    time_varying_beta: bool = False,
    constant_spend: bool = False
) -> Dict:
    """完全版DGP（詳細な貢献度追跡付き）+ Time-Varying Beta option"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if segment_attributes is None:
        segment_attributes = create_segment_attributes(3)
    
    K = segment_attributes.shape[0]
    T = n_days
    
    segment_params = generate_segment_params(segment_attributes)
    
    # 広告出稿量生成
    t_arr = np.arange(T)
    
    if constant_spend:
        # Constant Spend (Mean levels) + Small Noise
        tv_spend = np.full(T, 50.0) + np.random.randn(T) * 2
        display_spend = np.full(T, 30.0) + np.random.randn(T) * 2
        gen_search = np.full(T, 25.0) + np.random.randn(T) * 2
        brand_search = np.full(T, 20.0) + np.random.randn(T) * 2
        organic = np.full(T, 1.0)
    else:
        tv_spend = np.maximum(0, 50 + 30 * np.sin(2 * np.pi * t_arr / 365) + 
                              20 * np.sin(2 * np.pi * t_arr / 7) + np.random.randn(T) * 10)
        display_spend = np.maximum(0, 30 + 15 * np.cos(2 * np.pi * t_arr / 7) + np.random.randn(T) * 8)
        gen_search = np.maximum(0, 25 + np.random.randn(T) * 12)
        brand_search = np.maximum(0, 20 + np.random.randn(T) * 10)
        organic = 1 + 0.3 * np.sin(2 * np.pi * t_arr / 180)
    
    X = np.column_stack([tv_spend, display_spend, gen_search, brand_search, organic])
    
    # 正規化（Mean=1.0）
    X_mean = X.mean(axis=0) + 1e-8
    X_norm_input = X / X_mean
    
    # Hill Transform (Gamma=2.0, Kappa=0.5)
    gamma_true = 2.0
    kappa_true = 0.5
    X_transformed = hill_transform(X_norm_input, gamma=gamma_true, kappa=kappa_true)
    
    # Time-Varying Multipliers (Seasonality for Ads)
    # Shape: [T, 5] (TV, Disp, Gen, Brand, Organic)
    time_varying_multipliers = np.ones((T, 5))
    if time_varying_beta:
        # TV: Peaks in December (Winter)
        time_varying_multipliers[:, 0] = 1.0 + 0.3 * np.sin(2 * np.pi * t_arr / 365 - np.pi/2)
        # Display: Weak Seasonality
        time_varying_multipliers[:, 1] = 1.0 + 0.1 * np.cos(2 * np.pi * t_arr / 180)
    
    # 全セグメントをシミュレーション
    all_states = {}
    total_C_all = np.zeros(T)
    
    # セグメント別貢献度を集約
    total_attr = {
        'TV_awareness': 0, 'Display_awareness': 0, 'GenSearch_awareness': 0, 'Organic_awareness': 0,
        'Display_conversion': 0, 'GenSearch_conversion': 0, 'BrandSearch_conversion': 0, 'Organic_conversion': 0,
        'total_conversions': 0
    }
    
    for k in range(K):
        M_seg = market_size * segment_params[k]['pop_share']
        # 変換後の圧力を使用
        result = _simulate_segment_with_attribution(segment_params[k], X_transformed, M_seg, time_varying_multipliers)
        
        all_states[f'segment_{k}'] = result['states']
        total_C_all += result['states']['C']
        
        # 貢献度を累積
        for key in total_attr.keys():
            total_attr[key] += result['attribution'][key]
    
    # 最終的なチャネル貢献度（U→A + A→C の合計を100%に正規化）
    total_contribution = (
        total_attr['TV_awareness'] + 
        total_attr['Display_awareness'] + total_attr['Display_conversion'] +
        total_attr['GenSearch_awareness'] + total_attr['GenSearch_conversion'] +
        total_attr['BrandSearch_conversion'] +
        total_attr['Organic_awareness'] + total_attr['Organic_conversion']
    )
    
    true_attribution = {
        'TV': total_attr['TV_awareness'] / (total_contribution + 1e-8) * 100,
        'Display': (total_attr['Display_awareness'] + total_attr['Display_conversion']) / (total_contribution + 1e-8) * 100,
        'GenSearch': (total_attr['GenSearch_awareness'] + total_attr['GenSearch_conversion']) / (total_contribution + 1e-8) * 100,
        'BrandSearch': total_attr['BrandSearch_conversion'] / (total_contribution + 1e-8) * 100,
        'Organic': (total_attr['Organic_awareness'] + total_attr['Organic_conversion']) / (total_contribution + 1e-8) * 100,
        'total_conversions': total_attr['total_conversions'],
        'details': total_attr,
    }
    
    # 売上生成
    sales_clean = base_conversion + total_C_all
    noise = np.random.randn(T) * noise_scale * np.sqrt(sales_clean + 1)
    y = np.maximum(1, sales_clean + noise)
    
    return {
        'X': torch.tensor(X_norm_input, dtype=torch.float32), # 正規化入力を返す
        'y': torch.tensor(y, dtype=torch.float32),
        'segment_params': segment_params,
        'true_attribution': true_attribution,
        'states': all_states,
        'segment_attributes': segment_attributes,
        'channel_names': ['TV', 'Display', 'GenSearch', 'BrandSearch', 'Organic'],
        'time_varying_multipliers': time_varying_multipliers,
    }


def print_dgp_summary(data: Dict) -> None:
    """DGPデータのサマリーを表示"""
    print("="*60)
    print("完全版DGP (Normalized & Hill Transformed) サマリー")
    print("="*60)
    
    print(f"\nX: shape={data['X'].shape}, mean={data['X'].mean():.4f}")
    print(f"y: shape={data['y'].shape}, mean={data['y'].mean():.1f}")
    
    print("\n【真のチャネル貢献度】")
    attr = data['true_attribution']
    print(f"  TV:          {attr['TV']:.1f}%")
    print(f"  Display:     {attr['Display']:.1f}%")
    print(f"  GenSearch:   {attr['GenSearch']:.1f}%")
    print(f"  BrandSearch: {attr['BrandSearch']:.1f}%")
    print(f"  Organic:     {attr['Organic']:.1f}%")
    print(f"  総購買: {attr['total_conversions']:.0f}人")


if __name__ == '__main__':
    segment_attrs = create_segment_attributes(3)
    data = generate_dgp_data(n_days=730, seed=42, segment_attributes=segment_attrs)
    print_dgp_summary(data)
