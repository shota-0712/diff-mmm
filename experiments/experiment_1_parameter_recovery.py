#!/usr/bin/env python3
"""
実験1: パラメータ復元実験
========================

【目的】
論文 表11.7「パラメータ復元実験の結果」を生成。
DiffMMMがDGP（データ生成過程）の真のパラメータを正しく推定できるかを検証。
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.diff_mmm import DiffMMM, ModelConfig, Trainer
from dgp.state_space_dgp import generate_dgp_data, create_segment_attributes, print_dgp_summary


def run_parameter_recovery(n_trials=5):
    """
    パラメータ復元実験のメイン関数
    """
    print("="*70)
    print("実験1: パラメータ復元実験 (単一セグメント・逐次計算版)")
    print("="*70)
    
    # セグメント属性（固定3セグメント）
    segment_attrs = create_segment_attributes(3)
    
    # サンプルデータで真値を表示
    print("\n【Step 1】DGPの真値を確認")
    print("-"*70)
    data_sample = generate_dgp_data(n_days=730, seed=42, segment_attributes=segment_attrs)
    print_dgp_summary(data_sample)
    
    # 結果格納（セグメント別）
    results = {k: {
        'alpha_UA': [], 'beta_tv_UA': [], 'beta_disp_UA': [], 'beta_gen_UA': [],
        'lambda_k': [], 'beta_disp_AC': [],
        'alpha_AC': [], 'beta_gen_AC': [], 'beta_brand_AC': [],
    } for k in range(3)}
    
    print(f"\n【Step 2】DiffMMM学習と推定（{n_trials}試行）")
    print("-"*70)
    
    for i in range(n_trials):
        print(f"試行 {i+1}/{n_trials} ...", end=' ')
        sys.stdout.flush()
        
        seed = i + 200
        
        # データ生成
        data = generate_dgp_data(
            n_days=730,
            seed=seed,
            segment_attributes=segment_attrs
        )
        
        # 【デバッグ】単一チャネル（TV）のみにする
        # TV(idx 0)以外を0にする
        # data['X'][:, 1:] = 0.0
        # print("【デバッグモード】TV以外の出稿量を0にしました。単一チャネルでの復元を確認します。")
        
        # モデル設定
        config = ModelConfig(
            n_segments=3,
            n_states=3,
            n_features=3,
            use_paramnet=True,
            paramnet_hidden_dim=32,
            max_epochs=1500,  # エポック数を増加
            patience=100,
            learning_rate=0.005,
            gamma_fixed=2.0,  # 真の構造（Gamma=2.0）固定
            kappa_fixed=0.5   # 真の構造（Kappa=0.5）固定
        )
        
        # モデル学習
        model = DiffMMM(
            T=len(data['X']),
            config=config,
            segment_attributes=segment_attrs
        )
        
        trainer = Trainer(model, config)
        trainer.fit(data['X'], data['y'], verbose=False)
        
        # パラメータ取得
        model.eval()
        with torch.no_grad():
            params_dict = model.param_net(segment_attrs)
            
            # セグメント別に保存
            for k in range(3):
                results[k]['alpha_UA'].append(params_dict['alpha_UA'][k].item())
                results[k]['beta_tv_UA'].append(params_dict['beta_UA'][k, 0].item())     # TV
                results[k]['beta_disp_UA'].append(params_dict['beta_UA'][k, 1].item())   # Display
                results[k]['beta_gen_UA'].append(params_dict['beta_UA'][k, 2].item())    # GenSearch
                results[k]['lambda_k'].append(params_dict['lambda_k'][k].item())
                
                results[k]['beta_disp_AC'].append(params_dict['beta_AC_display'][k].item()) # Display A->C
                
                results[k]['alpha_AC'].append(params_dict['alpha_AC'][k].item())
                results[k]['beta_gen_AC'].append(params_dict['beta_AC'][k, 0].item())    # GenSearch
                results[k]['beta_brand_AC'].append(params_dict['beta_AC'][k, 1].item())  # BrandSearch
        
        print("完了")
    
    # ========================================
    # 結果出力
    # ========================================
    print("\n【Step 3】パラメータ復元結果（セグメント別）")
    print("="*70)
    
    for k in range(3):
        true_params = data_sample['segment_params'][k]
        attrs = segment_attrs[k].numpy()
        
        print(f"\n【セグメント{k+1}】(Age={attrs[0]:.2f}, Female={attrs[1]:.2f}, Urban={attrs[2]:.2f})")
        print(f"{'パラメータ':<16} {'真値':>8} {'推定値':>8} {'標準偏差':>8} {'相対誤差':>10}")
        print("-"*55)
        
        # Alphaも表示対象に追加
        for param_name in ['alpha_UA', 'beta_tv_UA', 'beta_disp_UA', 'beta_gen_UA', 'lambda_k', 'alpha_AC', 'beta_disp_AC', 'beta_gen_AC', 'beta_brand_AC']:
            if len(results[k][param_name]) == 0:
                continue
                
            true_val = true_params[param_name]
            est_mean = np.mean(results[k][param_name])
            est_std = np.std(results[k][param_name])
            
            # lambda_kの解釈（logit vs probability）の注意
            check_val = est_mean
            if param_name == 'lambda_k':
                    # モデルの出力はlogitなので確率に変換して比較
                    check_val = 1 / (1 + np.exp(-est_mean))
            
            if abs(true_val) > 0.01:
                rel_err = (check_val - true_val) / abs(true_val) * 100
                error_str = f"{rel_err:+9.1f}%"
            else:
                error_str = "---"
            
            print(f"{param_name:<16} {true_val:>8.4f} {check_val:>8.4f} {est_std:>8.4f} {error_str:>10}")
            
    return results, data_sample


if __name__ == '__main__':
    run_parameter_recovery(n_trials=5)
