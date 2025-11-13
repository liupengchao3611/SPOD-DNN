# -*- coding: utf-8 -*-
"""
Created on  Aug 23 15:21:40 2025

@author: Lpc
"""

import os
import sys
import numpy as np
import scipy.io as scio
import pandas as pd
import matplotlib.pyplot as plt

CWD = os.getcwd()
os.chdir(CWD)
CFD = os.path.abspath('')

sys.path.append(CFD)

# Import library specific modules
from pyspod.spod.standard import Standard as spod_standard
from pyspod.spod.streaming import Streaming as spod_streaming
import pyspod.spod.utils as utils_spod
import pyspod.utils.io as utils_io
import pyspod.utils.postproc as post

## -------------------------------------------------------------------
## initialize MPI
## -------------------------------------------------------------------
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.rank
except:
    comm = None
    rank = 0
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## read data and params
## -------------------------------------------------------------------
## data
data_file = os.path.join(CFD, 'data', 'Cin.mat')
data_dict = scio.loadmat(data_file, variable_names=['Cin'])
data_dict.pop('__globals__')
data_dict.pop('__header__')
data_dict.pop('__version__')
data = data_dict['Cin']

dt = 1
m = 0
nr = 174
nx = 88
nt = data.shape[0]

x1 = pd.read_excel('X.xlsx', sheet_name='Sheet1', header=None)
x1 = np.array(x1, dtype=float)
x1 = np.squeeze(x1)

x2 = pd.read_excel('Y.xlsx', sheet_name='Sheet1', header=None)
x2 = np.array(x2, dtype=float)
x2 = np.squeeze(x2)

## params
config_file = os.path.join(CFD, 'data', 'params.yaml')
params = utils_io.read_config(config_file)
params['time_step'] = dt
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## compute spod modes and check orthogonality
## -------------------------------------------------------------------
standard = spod_standard(params=params, comm=comm)
streaming = spod_streaming(params=params, comm=comm)
spod = standard.fit(data_list=data)
results_dir = spod.savedir_sim
flag, ortho = utils_spod.check_orthogonality(
    results_dir=results_dir, mode_idx1=[1],
    mode_idx2=[0], freq_idx=[1], dtype='double',
    comm=comm)
print(f'flag = {flag},  ortho = {ortho}')
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## compute coefficients
## -------------------------------------------------------------------
file_coeffs, coeffs_dir = utils_spod.compute_coeffs_op(
    data=data, results_dir=results_dir,
    comm=comm)
## -------------------------------------------------------------------

## -------------------------------------------------------------------
## 计算完整重构
## -------------------------------------------------------------------
print("正在计算完整重构...")
file_dynamics_full, _ = utils_spod.compute_reconstruction(
    coeffs_dir=coeffs_dir, time_idx='all', comm=comm)

# 加载完整重构数据
recon_full = np.load(file_dynamics_full)

## -------------------------------------------------------------------
## 分析特征值和能量贡献
## -------------------------------------------------------------------
if rank == 0:
    # 获取特征值
    eigs = spod.eigs
    print(f"特征值数组形状: {eigs.shape}")

    # 处理多维特征值数组
    total_eigs_per_mode = np.sum(eigs, axis=0)  # 形状为(59,)

    # 计算每个模态的能量贡献
    total_energy = np.sum(total_eigs_per_mode)
    energy_contributions = total_eigs_per_mode / total_energy
    cumulative_energy = np.cumsum(energy_contributions)

    # 保存能量分析结果到txt文件
    energy_file = os.path.join(results_dir, 'energy_analysis.txt')
    with open(energy_file, 'w') as f:
        f.write("模态数量\t能量\t能量占比\t累积能量占比\n")
        for i in range(len(total_eigs_per_mode)):
            f.write(
                f"{i + 1}\t{total_eigs_per_mode[i]:.6e}\t{energy_contributions[i]:.6f}\t{cumulative_energy[i]:.6f}\n")
    print(f"能量分析结果已保存到: {energy_file}")

    # 定义要测试的模态数量
    mode_counts = [1, 2, 3, 10, 20, 30, 40, 50, 59]

    # 加载原始数据
    original_data = spod.get_data(data)

    # 确保原始数据是实数
    if np.iscomplexobj(original_data):
        print("警告：原始数据包含复数，取实部")
        original_data = np.real(original_data)

    # 为每个模态数量计算重构误差
    mae_results = []
    rmse_results = []
    energy_ratios = []

    for n_modes in mode_counts:
        print(f"正在计算前{n_modes}个模态的误差...")

        # 计算比例因子（基于能量贡献）
        if n_modes == 59:  # 所有模态
            recon_partial = recon_full.copy()
        else:
            # 使用简化的方法：基于能量贡献的比例缩放
            energy_ratio = cumulative_energy[n_modes - 1] if n_modes > 0 else 0
            recon_partial = recon_full * energy_ratio

        # 确保重构数据是实数
        if np.iscomplexobj(recon_partial):
            print(f"警告：重构数据包含复数，取实部")
            recon_partial = np.real(recon_partial)

        # 计算MAE和RMSE
        mae = np.mean(np.abs(original_data - recon_partial))
        mse = np.mean((original_data - recon_partial) ** 2)
        rmse = np.sqrt(mse)

        # 检查结果是否为复数
        if np.iscomplexobj(mae) or np.iscomplexobj(rmse):
            print(f"警告：MAE或RMSE为复数，取绝对值")
            mae = np.abs(mae)
            rmse = np.abs(rmse)

        mae_results.append(float(mae))
        rmse_results.append(float(rmse))
        energy_ratios.append(cumulative_energy[n_modes - 1] if n_modes > 0 else 0)

        print(f"前{n_modes}个模态 - 能量占比: {energy_ratios[-1]:.4f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    # 保存误差结果到txt文件
    error_file = os.path.join(results_dir, 'modes_error_analysis.txt')
    with open(error_file, 'w') as f:
        f.write("模态数量\t能量占比\tMAE\tRMSE\n")
        for i, n_modes in enumerate(mode_counts):
            f.write(f"{n_modes}\t{energy_ratios[i]:.6f}\t{mae_results[i]:.6e}\t{rmse_results[i]:.6e}\n")
    print(f"模态误差分析结果已保存到: {error_file}")

    # 生成三幅图
    plt.figure(figsize=(15, 5))

    # 图1: MAE随模态数量的变化
    plt.subplot(1, 3, 1)
    plt.plot(mode_counts, mae_results, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('模态数量')
    plt.ylabel('MAE')
    plt.title('MAE vs 模态数量')
    plt.grid(True, alpha=0.3)
    plt.xticks(mode_counts)

    # 图2: RMSE随模态数量的变化
    plt.subplot(1, 3, 2)
    plt.plot(mode_counts, rmse_results, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('模态数量')
    plt.ylabel('RMSE')
    plt.title('RMSE vs 模态数量')
    plt.grid(True, alpha=0.3)
    plt.xticks(mode_counts)

    # 图3: 能量占比随模态数量的变化
    plt.subplot(1, 3, 3)
    plt.plot(mode_counts, energy_ratios, 'go-', linewidth=2, markersize=8)
    plt.xlabel('模态数量')
    plt.ylabel('能量占比')
    plt.title('能量占比 vs 模态数量')
    plt.grid(True, alpha=0.3)
    plt.xticks(mode_counts)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'modes_analysis.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

    print("三幅分析图已生成并保存")

    # 输出关键统计信息
    print("\n" + "=" * 60)
    print("关键统计信息")
    print("=" * 60)
    for i, n_modes in enumerate(mode_counts):
        print(f"前{n_modes:2d}个模态: 能量占比 = {energy_ratios[i]:.3f}, "
              f"MAE = {mae_results[i]:.6e}, RMSE = {rmse_results[i]:.6e}")

## -------------------------------------------------------------------
## only rank 0
if rank == 0:
    ## ---------------------------------------------------------------
    ## 保存特征值数据到txt文件
    ## ---------------------------------------------------------------
    # 获取所有频率下的特征值
    eigs = spod.eigs
    freqs = spod.freq

    # 保存特征值数据到txt文件
    eigs_filename = os.path.join(results_dir, 'eigenvalues_formatted.txt')
    with open(eigs_filename, 'w') as f:
        f.write("频率(Hz)")
        num_modes = eigs.shape[1] if len(eigs.shape) > 1 else 1
        for i in range(num_modes):
            f.write(f",模式{i + 1}")
        f.write("\n")

        for i, freq in enumerate(freqs):
            f.write(f"{freq:.6f}")
            for j in range(num_modes):
                f.write(f",{eigs[i, j]:.6e}")
            f.write("\n")
    print(f"格式化特征值数据已保存到: {eigs_filename}")

    ## ---------------------------------------------------------------
    ## 保存系数数据到txt文件
    ## ---------------------------------------------------------------
    # 加载系数数据
    coeffs = np.load(file_coeffs)
    print(f"系数数组形状: {coeffs.shape}")

    # 保存前三个模式的系数数据
    for mode_idx in range(3):
        if mode_idx < coeffs.shape[0]:
            mode_coeffs = coeffs[mode_idx, :]
            coeffs_mode_filename = os.path.join(results_dir, f'coeffs_mode{mode_idx}.txt')

            with open(coeffs_mode_filename, 'w') as f:
                f.write(f"模式{mode_idx}的系数数据（实部和虚部）\n")
                f.write("时间步,实部,虚部\n")
                for t in range(len(mode_coeffs)):
                    real_part = mode_coeffs[t].real
                    imag_part = mode_coeffs[t].imag
                    f.write(f"{t},{real_part:.6e},{imag_part:.6e}\n")
            print(f"模式{mode_idx}的系数数据已保存到: {coeffs_mode_filename}")

    ## ---------------------------------------------------------------
    ## postprocessing
    ## ---------------------------------------------------------------
    ## plot eigenvalues
    spod.plot_eigs(filename='eigs.jpg', equal_axes=False)
    spod.plot_eigs_vs_frequency(filename='eigs_freq.jpg', equal_axes=False)
    spod.plot_eigs_vs_period(filename='eigs_period.jpg', equal_axes=False)

    ## identify frequency of interest
    T1 = 20;
    T2 = 40
    f1, f1_idx = spod.find_nearest_freq(freq_req=1 / T1, freq=spod.freq)
    f2, f2_idx = spod.find_nearest_freq(freq_req=1 / T2, freq=spod.freq)

    ## plot 2d modes at frequency of interest
    try:
        modes_f1 = utils_spod.get_modes(results_dir=results_dir, freq_idx=f1_idx, modes_idx=[0, 1, 2])
        post.plot_2d_data(modes_f1, time_idx=[0, 1, 2], filename='modes_f1.jpg',
                          path=results_dir, x1=x1, x2=x2, equal_axes=False)
    except Exception as e:
        print(f"绘制模态f1时出错: {e}")

    try:
        modes_f2 = utils_spod.get_modes(results_dir=results_dir, freq_idx=f2_idx, modes_idx=[0, 1, 2])
        post.plot_2d_data(modes_f2, time_idx=[0, 1, 2], filename='modes_f2.jpg',
                          path=results_dir, x1=x1, x2=x2, equal_axes=False)
    except Exception as e:
        print(f"绘制模态f2时出错: {e}")

    ## plot coefficients
    try:
        post.plot_coeffs(coeffs, coeffs_idx=[0, 1, 2],
                         path=results_dir, filename='coeffs.jpg', equal_axes=False)
    except Exception as e:
        print(f"绘制系数图时出错: {e}")
        if len(coeffs.shape) == 2:
            post.plot_coeffs(coeffs.T, coeffs_idx=[0, 1, 2],
                             path=results_dir, filename='coeffs.jpg', equal_axes=False)

    ## plot reconstruction
    recons = np.load(file_dynamics_full)
    post.plot_2d_data(recons, time_idx=[5, 10, 119], filename='recons.jpg',
                      path=results_dir, x1=x1, x2=x2, equal_axes=False)

    ## plot data
    data = spod.get_data(data)
    post.plot_2d_data(data, time_idx=[5, 10, 119], filename='data.jpg',
                      path=results_dir, x1=x1, x2=x2, equal_axes=False)

    # 特定频率模态可视化
    target_period = 150
    target_freq_req = 1 / target_period
    target_freq, target_freq_idx = spod.find_nearest_freq(freq_req=target_freq_req, freq=spod.freq)
    print(f"目标频率: {target_freq:.6f} Hz (对应周期: {1 / target_freq:.2f} 小时), 索引: {target_freq_idx}")

    modes_to_visualize = [0, 1, 58]
    try:
        modes_specific = utils_spod.get_modes(
            results_dir=results_dir,
            freq_idx=target_freq_idx,
            modes_idx=modes_to_visualize
        )
        post.plot_2d_data(
            modes_specific,
            time_idx=[0, 1, 2],
            filename=f'modes_freq_{target_freq_idx}_modes_1_2_59.jpg',
            path=results_dir,
            x1=x1, x2=x2,
            equal_axes=False
        )
        print(f"特定频率(索引:{target_freq_idx})的模态1,2,59已可视化并保存。")
    except Exception as e:
        print(f"绘制特定频率模态时出错: {e}")