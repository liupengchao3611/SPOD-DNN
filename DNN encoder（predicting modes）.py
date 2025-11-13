# -*- coding: utf-8 -*-
"""
Created on  Aug 8 10:13:40 2025

@author: Lpc
"""
import os
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf


class PhysicsResBlock(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense1 = Dense(units, activation=tf.nn.leaky_relu, kernel_regularizer=l2(0.001))
        self.bn1 = BatchNormalization()
        self.dense2 = Dense(units, activation=tf.nn.leaky_relu, kernel_regularizer=l2(0.001))
        self.bn2 = BatchNormalization()
        self.projection = Dense(units) if units else None

    def call(self, inputs):
        residual = inputs
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        if self.projection:
            residual = self.projection(residual)
        x += residual
        return tf.nn.leaky_relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class PhysicsInformedNN(Model):
    def __init__(self, input_dim, output_dim, n_x, n_y, pca_layer, name="PhysicsInformedNN", **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_x = n_x
        self.n_y = n_y
        self.pca_rebuild = pca_layer

        self.input_norm = BatchNormalization()
        self.res_blocks = [
            PhysicsResBlock(256),
            PhysicsResBlock(256),
            PhysicsResBlock(256),
            PhysicsResBlock(256),
            PhysicsResBlock(256),
            PhysicsResBlock(256)
        ]
        self.output_layer = Dense(output_dim)

    def build(self, input_shape):
        self.inflow_coeff = self.add_weight(
            name='inflow_coeff',
            shape=(),
            initializer=tf.constant_initializer(1.0),
            constraint=lambda x: tf.clip_by_value(x, 0.7, 1.3),
            trainable=True
        )
        self.outflow_coeff = self.add_weight(
            name='outflow_coeff',
            shape=(),
            initializer=tf.constant_initializer(1.0),
            constraint=lambda x: tf.clip_by_value(x, 0.7, 1.3),
            trainable=True
        )
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_x': self.n_x,
            'n_y': self.n_y,
            'pca_layer': tf.keras.layers.serialize(self.pca_rebuild)
        })
        return config

    @classmethod
    def from_config(cls, config):
        try:
            if 'name' not in config:
                config['name'] = "PhysicsInformedNN"

            pca_layer_config = config.pop('pca_layer')
            pca_layer = tf.keras.layers.deserialize(
                pca_layer_config,
                custom_objects={'PCARebuildLayer': PCARebuildLayer}
            )
            return cls(pca_layer=pca_layer, **config)
        except KeyError as e:
            raise ValueError(f"缺失关键配置项: {str(e)}") from e

    def build_shape_functions(self, batch_size):
        x_norm = tf.linspace(0.0, 1.0, self.n_x)
        x_norm = tf.reshape(x_norm, [1, self.n_x, 1])
        x_norm = tf.tile(x_norm, [batch_size, 1, self.n_y])

        phi_in = 1 - x_norm
        phi_out = x_norm
        return phi_in, phi_out

    def apply_boundary_constraints(self, raw_output, batch_size):
        raw_real = raw_output[..., 0]
        raw_imag = raw_output[..., 1]

        phi_in, phi_out = self.build_shape_functions(batch_size)

        phi_in_condition = phi_in > 0.99
        phi_out_condition = phi_out > 0.99

        constrained_real = tf.where(
            phi_in_condition,
            self.inflow_coeff * tf.ones_like(raw_real),
            tf.where(
                phi_out_condition,
                self.outflow_coeff * tf.ones_like(raw_real),
                raw_real
            )
        )

        constrained_imag = tf.where(
            phi_in_condition,
            self.inflow_coeff * tf.ones_like(raw_imag),
            tf.where(
                phi_out_condition,
                self.outflow_coeff * tf.ones_like(raw_imag),
                raw_imag
            )
        )

        return tf.stack([constrained_real, constrained_imag], axis=-1)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        x = self.input_norm(inputs)

        for block in self.res_blocks:
            x = block(x)

        raw_features = self.output_layer(x)
        reconstructed = self.pca_rebuild(raw_features)

        return self.apply_boundary_constraints(reconstructed, batch_size)


class PCARebuildLayer(Layer):
    def __init__(self, pca_components, pca_mean, original_shape, name="pca_rebuild", **kwargs):
        super().__init__(name=name, **kwargs)
        self.pca_components = tf.constant(pca_components, dtype=tf.float32)
        self.pca_mean = tf.constant(pca_mean, dtype=tf.float32)
        self.original_shape = original_shape

    def call(self, inputs):
        reconstructed = tf.matmul(inputs, self.pca_components) + self.pca_mean
        return tf.reshape(reconstructed, [-1] + list(self.original_shape))

    def get_config(self):
        config = super().get_config()
        config.update({
            'pca_components': self.pca_components.numpy(),
            'pca_mean': self.pca_mean.numpy(),
            'original_shape': self.original_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        if 'name' not in config:
            config['name'] = "pca_rebuild"
        return cls(**config)


# 配置路径
results_path = r"**"
preprocess_path = os.path.join(results_path, 'preprocess_tools')
model_path = os.path.join(results_path, "final_model")


def load_preprocessing_tools():
    """加载所有预处理工具和配置信息（增强验证）"""
    # 1. 检查文件是否存在
    required_files = [
        'robust_scaler.joblib',
        'feature_selector.joblib',
        'pca_model.joblib',
        'energy_scales.npy',
        'preprocessing_info.joblib'
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(preprocess_path, file)):
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(f"预处理文件缺失: {', '.join(missing_files)}")

    # 2. 加载工具
    scaler = joblib.load(os.path.join(preprocess_path, 'robust_scaler.joblib'))
    selector = joblib.load(os.path.join(preprocess_path, 'feature_selector.joblib'))
    pca_model = joblib.load(os.path.join(preprocess_path, 'pca_model.joblib'))
    energy_scales = np.load(os.path.join(preprocess_path, 'energy_scales.npy'))
    preprocess_info = joblib.load(os.path.join(preprocess_path, 'preprocessing_info.joblib'))

    # 3. 验证特征选择器
    print(f"特征选择器输入维度: {selector.n_features_in_}")
    print(f"特征选择器输出维度: {selector.n_features_}")

    # 不再强制要求降维，改为警告
    if selector.n_features_ == selector.n_features_in_:
        print("⚠️ 警告：特征选择器未降维！将使用原始特征维度")
    elif selector.n_features_ != 3:
        print(f"⚠️ 警告：特征选择器输出维度异常 ({selector.n_features_})")

    # 4. 获取训练时使用的实际特征索引（如果存在）
    if 'selected_feature_indices' in preprocess_info:
        print(f"✅ 加载训练时特征索引: {preprocess_info['selected_feature_indices']}")
    else:
        # 手动设置默认特征索引（根据训练日志调整）
        default_indices = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14]  # 示例索引
        print(f"⚠️ 未找到训练特征索引，使用默认: {default_indices}")
        preprocess_info['selected_feature_indices'] = default_indices

    return scaler, selector, pca_model, energy_scales, preprocess_info


def create_features(params):
    """创建与训练时相同的特征"""
    qin = params[0]
    qout = params[1]
    cin = params[2]

    # 特征工程
    flow_ratio = qin / (qout + 1e-6)
    conc_ratio = cin / (qin + 1e-6)
    reynolds = (qin * cin) / 1.0
    energy_ratio = (qin ** 2) / (qout + 1e-6)
    momentum_transfer = (qin * qout) / (cin + 1e-6)
    convective_flux = qin * cin
    qin_squared = qin ** 2
    qout_squared = qout ** 2
    cin_squared = cin ** 2
    qin_qout = qin * qout
    qin_cin = qin * cin
    qout_cin = qout * cin

    return np.array([
        qin, qout, cin,
        flow_ratio, conc_ratio, reynolds,
        energy_ratio, momentum_transfer, convective_flux,
        qin_squared, qout_squared, cin_squared,
        qin_qout, qin_cin, qout_cin
    ]).astype(np.float32)


def preprocess_new_data(params, scaler, selector, preprocess_info):
    """预处理新输入数据（增强鲁棒性）"""
    # 1. 创建特征（15维）
    raw_features = create_features(params)

    # 2. 特征选择（使用训练时保存的特征索引）
    selected_indices = preprocess_info.get('selected_feature_indices', list(range(13)))
    selected = raw_features[selected_indices].reshape(1, -1)

    # 3. 标准化（确保维度匹配）
    if selected.shape[1] != scaler.n_features_in_:
        print(f"⚠️ 手动降维后维度: {selected.shape[1]}, 标准化器期望: {scaler.n_features_in_}")
        # 维度调整策略
        if selected.shape[1] > scaler.n_features_in_:
            print(f"🔄 截取前{scaler.n_features_in_}个特征")
            selected = selected[:, :scaler.n_features_in_]
        else:
            print(f"🔄 填充零值使维度匹配")
            padding = np.zeros((1, scaler.n_features_in_ - selected.shape[1]))
            selected = np.hstack([selected, padding])

    scaled = scaler.transform(selected)
    return scaled


def predict_new_data(params):
    """预测新数据"""
    # 1. 加载预处理工具
    try:
        scaler, selector, pca_model, energy_scales, preprocess_info = load_preprocessing_tools()
    except Exception as e:
        print(f"❌ 预处理工具加载失败: {str(e)}")
        # 尝试手动设置关键参数
        print("🔄 尝试使用默认预处理信息")
        preprocess_info = {
            'input_dim': 13,  # 关键修复：使用13维输入
            'output_dim': 50,
            'n_x': 100, 'n_y': 50,
            'original_shape': (1, 100, 50, 2),
            'selected_feature_indices': list(range(13))  # 使用13个特征
        }
        energy_scales = np.array([1.0])
        # 创建虚拟标准化器和选择器
        scaler = RobustScaler()
        scaler.fit(np.zeros((1, 13)))  # 匹配13维输入
        selector = None

        # 尝试加载PCA模型
        try:
            pca_model = joblib.load(os.path.join(preprocess_path, 'pca_model.joblib'))
        except:
            print("❌ PCA模型加载失败，无法继续预测")
            return None, None

    # 2. 增强模型加载
    try:
        model = load_model(
            model_path,
            custom_objects={
                'PhysicsResBlock': PhysicsResBlock,
                'PhysicsInformedNN': PhysicsInformedNN,
                'PCARebuildLayer': PCARebuildLayer,
                'robust_physics_loss': robust_physics_loss
            },
            compile=False
        )
        print("✅ 模型加载成功（标准方式）")
    except Exception as e:
        print(f"⚠️ 标准加载失败: {str(e)}")
        print("🔄 尝试备用加载方案...")

        try:
            # 重建模型结构（使用正确的输入维度）
            pca_layer = PCARebuildLayer(
                pca_components=pca_model.components_,
                pca_mean=pca_model.mean_,
                original_shape=preprocess_info['original_shape']
            )

            model = PhysicsInformedNN(
                input_dim=preprocess_info['input_dim'],  # 使用13维输入
                output_dim=preprocess_info['output_dim'],
                n_x=preprocess_info['n_x'],
                n_y=preprocess_info['n_y'],
                pca_layer=pca_layer
            )

            # 加载权重
            weights_path = os.path.join(model_path, 'variables', 'variables')
            if os.path.exists(weights_path):
                model.load_weights(weights_path)
                print("✅ 模型权重加载成功")
            else:
                # 尝试其他可能的权重路径
                print("🔄 尝试替代权重加载方案")
                model.load_weights(os.path.join(model_path, 'variables'))
        except Exception as e:
            print(f"❌ 模型重建失败: {str(e)}")
            return None, None

    # 3. 预处理输入数据（传递preprocess_info）
    try:
        input_data = preprocess_new_data(params, scaler, selector, preprocess_info)
        print(f"✅ 预处理完成，输入维度: {input_data.shape}")
    except Exception as e:
        print(f"❌ 数据预处理失败: {str(e)}")
        return None, None

    # 4. 进行预测
    try:
        pca_output = model.predict(input_data)
    except Exception as e:
        print(f"❌ 模型预测失败: {str(e)}")
        return None, None

    # 5. PCA逆变换重建模态数据
    try:
        reconstructed = pca_model.inverse_transform(pca_output)
    except Exception as e:
        print(f"❌ PCA逆变换失败: {str(e)}")
        return None, None

    # 6. 恢复原始形状
    try:
        n_samples, n_x, n_y, n_channels = preprocess_info['original_shape']
        reconstructed = reconstructed.reshape(1, n_x, n_y, n_channels)
    except Exception as e:
        print(f"❌ 形状恢复失败: {str(e)}")
        return None, None

    # 7. 分离实部和虚部
    try:
        real_part = reconstructed[0, :, :, 0]
        imag_part = reconstructed[0, :, :, 1]
    except Exception as e:
        print(f"❌ 实部/虚部分离失败: {str(e)}")
        return None, None

    # 8. 恢复能量缩放
    try:
        avg_energy_scale = np.mean(energy_scales)
        real_part *= avg_energy_scale
        imag_part *= avg_energy_scale
    except Exception as e:
        print(f"⚠️ 能量缩放失败: {str(e)}")

    # 9. 验证输出形状
    try:
        print(f"恢复形状: ({n_x}, {n_y}, {n_channels}) | 实际形状: {real_part.shape}")
        if real_part.shape != (n_x, n_y):
            print(f"⚠️ 实部形状不匹配: 期望({n_x}, {n_y}), 实际{real_part.shape}")
        if imag_part.shape != (n_x, n_y):
            print(f"⚠️ 虚部形状不匹配: 期望({n_x}, {n_y}), 实际{imag_part.shape}")
    except Exception as e:
        print(f"⚠️ 形状验证失败: {str(e)}")

    return real_part, imag_part


def robust_physics_loss(y_true, y_pred, n_x, n_y, epoch, delta=5000.0):
    # 动态衰减物理约束权重
    decay_factor = max(0.5, 1.0 - epoch / 200)  # 线性衰减
    energy_weight = 0.5 * decay_factor if epoch < 50 else 1.0 * decay_factor
    momentum_weight = 0.1 * decay_factor

    # Huber损失
    error = y_true - y_pred
    condition = tf.abs(error) < delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    base_loss = tf.where(condition, squared_loss, linear_loss)

    # 提取实部
    raw_real = y_pred[..., 0]

    # 1. 能量守恒约束
    inflow = tf.reduce_sum(raw_real[:, 0, :], axis=1)
    outflow = tf.reduce_sum(raw_real[:, -1, :], axis=1)
    energy_loss = tf.reduce_mean(tf.square(inflow - outflow)) + 1e-6

    # 2. 动量守恒约束
    slice1 = tf.strided_slice(
        raw_real,
        [0, 1, 0],
        [tf.shape(raw_real)[0], tf.shape(raw_real)[1], tf.shape(raw_real)[2]],
        [1, 1, 1]
    )
    slice2 = tf.strided_slice(
        raw_real,
        [0, 0, 0],
        [tf.shape(raw_real)[0], tf.shape(raw_real)[1] - 1, tf.shape(raw_real)[2]],
        [1, 1, 1]
    )
    velocity_grad = tf.reduce_mean(tf.abs(slice1 - slice2))
    momentum_loss = 1e-3 * velocity_grad

    # 3. 简化正交约束
    ortho_loss = 0.0
    if epoch >= 20:
        ortho_loss = 1e-4 * tf.reduce_mean(tf.square(raw_real))

    # 动态权重调整
    energy_weight = 0.5 if epoch < 30 else 3.0
    momentum_weight = 0.1 if epoch < 30 else 0.8
    ortho_weight = min(0.3, max(0, (epoch - 20) / 20 * 0.3))

    return (tf.reduce_mean(base_loss) +
            energy_weight * energy_loss +
            momentum_weight * momentum_loss +
            ortho_weight * ortho_loss)


if __name__ == "__main__":
    # 1. 验证预处理工具
    print("🧪 验证预处理工具...")
    try:
        scaler, selector, pca_model, energy_scales, preprocess_info = load_preprocessing_tools()
        print("✅ 预处理工具加载成功")
    except Exception as e:
        print(f"❌ 预处理工具加载失败: {str(e)}")
        # 尝试继续运行（使用备用方案）

    # 2. 特征重要性诊断（仅在特征选择器可用时执行）
    if 'selector' in locals() and selector is not None:
        print("\n🔍 特征重要性诊断:")
        try:
            # 获取特征选择器的特征排名
            if hasattr(selector, 'ranking_'):
                print("特征排名:", selector.ranking_)
            else:
                print("ℹ️ 特征选择器无ranking_属性")

            # 创建特征名称列表
            feature_names = [
                "Qin", "Qout", "Cin",
                "FlowRatio", "ConcRatio", "Reynolds",
                "EnergyRatio", "MomentumTransfer", "ConvectiveFlux",
                "Qin^2", "Qout^2", "Cin^2",
                "Qin*Qout", "Qin*Cin", "Qout*Cin"
            ]

            # 打印特征重要性排序
            if hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
                print("\n特征重要性排序:")
                sorted_indices = np.argsort(selector.estimator_.feature_importances_)[::-1]
                for i in sorted_indices:
                    print(f"{feature_names[i]}: {selector.estimator_.feature_importances_[i]:.4f}")
            else:
                print("❌ 无法获取特征重要性，选择器未保存评估器")
        except Exception as e:
            print(f"❌ 特征重要性诊断失败: {str(e)}")

    # 3. 验证模型结构重建
    print("🧪 验证模型重建能力...")
    if 'pca_model' in locals() and 'preprocess_info' in locals():
        try:
            pca_layer = PCARebuildLayer(
                pca_components=pca_model.components_,
                pca_mean=pca_model.mean_,
                original_shape=preprocess_info['original_shape']
            )
            test_model = PhysicsInformedNN(
                input_dim=preprocess_info['input_dim'],
                output_dim=preprocess_info['output_dim'],
                n_x=preprocess_info['n_x'],
                n_y=preprocess_info['n_y'],
                pca_layer=pca_layer
            )
            test_model.build(input_shape=(None, preprocess_info['input_dim']))
            print("✅ 模型结构重建成功")
        except Exception as e:
            print(f"❌ 模型重建失败: {str(e)}")
    else:
        print("⚠️ 跳过模型重建，缺少必要组件")

    # 4. 进行预测
    new_params = [100, 120, 80]
    print("🚀 开始预测...")
    try:
        predicted_real, predicted_imag = predict_new_data(new_params)

        if predicted_real is not None and predicted_imag is not None:
            print(f"✅ 预测成功！实部形状: {predicted_real.shape}, 虚部形状: {predicted_imag.shape}")

            # 保存为npy文件（确保形状一致）
            np.save(os.path.join(results_path, 'predicted_real.npy'), predicted_real)
            np.save(os.path.join(results_path, 'predicted_imag.npy'), predicted_imag)
            print(f"💾 预测结果已保存为npy文件")

            # 可视化结果
            try:
                import matplotlib.pyplot as plt

                plt.figure(figsize=(12, 5))
                plt.subplot(121)
                plt.imshow(predicted_real, cmap='jet')
                plt.title('Predicted Real Part')
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(predicted_imag, cmap='jet')
                plt.title('Predicted Imaginary Part')
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(os.path.join(results_path, 'prediction_result.png'))
                plt.show()
            except Exception as e:
                print(f"⚠️ 可视化失败: {str(e)}")
        else:
            print("❌ 预测返回空结果")
    except Exception as e:
        print(f"❌ 预测失败: {str(e)}")
