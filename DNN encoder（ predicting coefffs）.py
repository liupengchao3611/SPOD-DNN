# -*- coding: utf-8 -*-
"""
Created on  Aug 16 15:21:40 2025

@author: Lpc
"""
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from scipy.interpolate import CubicSpline


# 定义自定义损失函数
class ComplexMSELoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        n = tf.shape(y_true)[-1] // 2
        real_true, imag_true = y_true[..., :n], y_true[..., n:]
        real_pred, imag_pred = y_pred[..., :n], y_pred[..., n:]
        real_loss = tf.reduce_mean(tf.square(real_true - real_pred))
        imag_loss = tf.reduce_mean(tf.square(imag_true - imag_pred))
        return real_loss + imag_loss

    def get_config(self):
        return super().get_config()


class PhysicsConstrainedLoss(tf.keras.losses.Loss):
    def __init__(self, base_loss, alpha=0.1, **kwargs):
        super().__init__(**kwargs)
        self.base_loss = base_loss
        self.alpha = alpha

    def call(self, y_true, y_pred):
        main_loss = self.base_loss(y_true, y_pred)
        conc_pred = self.reconstruct_concentration(y_pred)
        conc_pred_pos = tf.nn.relu(conc_pred)
        neg_loss = tf.reduce_mean(tf.square(conc_pred - conc_pred_pos))
        return main_loss + self.alpha * neg_loss

    def reconstruct_concentration(self, pred):
        return pred

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"base_loss": self.base_loss, "alpha": self.alpha})
        return base_config


def predict_coeffs(qin, qout, cin, results_path, mode_index=0):
    """
    使用训练好的模型预测系数，并提取特定模态的时间序列

    参数:
    qin, qout, cin: 输入参数 (Qin, Qout, Cin)
    results_path: 存储模型和预处理文件的目录路径
    mode_index: 要提取的模态索引 (默认0)

    返回:
    coeffs_selected: 指定模态的系数时间序列 (1, 120) 复数形式
    """
    # 确保模型和预处理文件存在
    required_files = [
        'final_coeffs_model.keras',
        'coeffs_scaler.pkl',
        'poly_features.pkl',
        'scaler.pkl'
    ]

    for file in required_files:
        file_path = os.path.join(results_path, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ 找不到必需文件: {file_path}")

    # 加载预处理对象
    scaler = joblib.load(os.path.join(results_path, 'scaler.pkl'))
    poly = joblib.load(os.path.join(results_path, 'poly_features.pkl'))
    coeffs_scaler = joblib.load(os.path.join(results_path, 'coeffs_scaler.pkl'))

    # 加载模型 (包含自定义对象)
    custom_objects = {
        'ComplexMSELoss': ComplexMSELoss,
        'PhysicsConstrainedLoss': PhysicsConstrainedLoss
    }
    model = load_model(
        os.path.join(results_path, 'final_coeffs_model.keras'),
        custom_objects=custom_objects
    )

    # 准备输入数据
    params_extended = poly.transform([[qin, qout, cin]])
    params_norm = scaler.transform(params_extended)

    # 预测
    pred_coeffs_flat = model.predict(params_norm, verbose=0)

    # 逆标准化
    pred_coeffs_flat = coeffs_scaler.inverse_transform(pred_coeffs_flat)

    # 获取预测的点数
    total_points = pred_coeffs_flat.shape[1] // 2  # 因为分成了实部和虚部
    n_modes = 8

    # 计算模型预测的时间步数
    predicted_time_steps = total_points // n_modes

    # 重构为复数矩阵 (原始形状)
    coeffs_real = pred_coeffs_flat[0, :total_points].reshape(n_modes, predicted_time_steps)
    coeffs_imag = pred_coeffs_flat[0, total_points:].reshape(n_modes, predicted_time_steps)

    # 创建插值函数，扩展到 120 个时间步
    coeffs_real_interp = np.zeros((n_modes, 120))
    coeffs_imag_interp = np.zeros((n_modes, 120))

    # 原始时间点
    original_x = np.linspace(0, 1, predicted_time_steps)

    # 新时间点
    new_x = np.linspace(0, 1, 120)

    # 对每个模式应用三次样条插值
    for i in range(n_modes):
        # 实部插值
        cs_real = CubicSpline(original_x, coeffs_real[i, :])
        coeffs_real_interp[i, :] = cs_real(new_x)

        # 虚部插值
        cs_imag = CubicSpline(original_x, coeffs_imag[i, :])
        coeffs_imag_interp[i, :] = cs_imag(new_x)

    # 重新组合为复数矩阵
    coeffs_predicted = coeffs_real_interp + 1j * coeffs_imag_interp

    # 提取指定模态的时间序列
    coeffs_selected = coeffs_predicted[mode_index:mode_index + 1, :]

    print(f"✅ 完整预测维度: {coeffs_predicted.shape} (模式×时间步)")
    print(f"✅ 提取模态索引: {mode_index}，输出形状: {coeffs_selected.shape}")

    return coeffs_selected


if __name__ == "__main__":
    # 配置参数
    results_path = r" target file path "  # 结果目录路径

    # 新预测参数
    qin_new = *
    qout_new = *
    cin_new = *

    # 要提取的模态索引 (0-7)
    target_mode = 0

    try:
        # 执行预测
        predicted_coeffs = predict_coeffs(qin_new, qout_new, cin_new, results_path, target_mode)

        # 保存结果
        output_file = f"predicted_coeffs_{qin_new}{qout_new}-{cin_new}_mode{target_mode}.npy"
        np.save(os.path.join(results_path, output_file), predicted_coeffs)
        print(f"✅ 预测结果已保存到: {os.path.join(results_path, output_file)}")

        # 打印结果验证
        print("\n预测的系数时间序列 (前10个时间点):")
        print(predicted_coeffs[0, :10])  # 注意现在形状是(1,120)

    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")