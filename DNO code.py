import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Layer, Input
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from scipy.fft import fft, fftfreq, ifft
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from itertools import product
import matplotlib
import joblib
from sklearn.linear_model import LassoCV
from sklearn.utils import resample

# 修复字体问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ======================== 配置文件路径 ========================
folder_path = r"E:\wdgmodes\SPOD（第二模态）"
results_path = r"E:\wdgmodes\SPOD（第二模态）\results"
os.makedirs(results_path, exist_ok=True)


# ======================== 基于解空间的约束类 ========================
class SolutionSpaceConstraint:
    """基于训练样本解空间的约束机制"""

    def __init__(self, training_params, training_modes_real, training_modes_imag):
        self.training_params = training_params[:, :3]  # 只取前3维
        self.training_modes_real = training_modes_real
        self.training_modes_imag = training_modes_imag

        self._build_parameter_space()
        self._compute_training_statistics()
        self._build_nearest_neighbors()

        print(f"✅ 解空间约束初始化完成 - 样本数: {len(self.training_params)}")

    def _build_parameter_space(self):
        """构建参数空间索引"""
        self.param_to_idx = {}
        for i, (qin, qout, cin) in enumerate(self.training_params):
            key = (int(qin), int(qout), int(cin))
            self.param_to_idx[key] = i

        self.unique_qin = np.unique(self.training_params[:, 0])
        self.unique_qout = np.unique(self.training_params[:, 1])
        self.unique_cin = np.unique(self.training_params[:, 2])

    def _compute_training_statistics(self):
        """计算训练数据的统计特征"""
        self.energy_stats = {
            'mean': np.mean(np.abs(self.training_modes_real) ** 2 + np.abs(self.training_modes_imag) ** 2, axis=0),
            'std': np.std(np.abs(self.training_modes_real) ** 2 + np.abs(self.training_modes_imag) ** 2, axis=0),
        }

        self.real_stats = {
            'mean': np.mean(self.training_modes_real, axis=0),
            'std': np.std(self.training_modes_real, axis=0)
        }
        self.imag_stats = {
            'mean': np.mean(self.training_modes_imag, axis=0),
            'std': np.std(self.training_modes_imag, axis=0)
        }

    def _build_nearest_neighbors(self):
        """构建最近邻搜索器"""
        self.nn_searcher = NearestNeighbors(n_neighbors=5, metric='euclidean')
        self.nn_searcher.fit(self.training_params)

    def extract_original_params(self, features):
        """从特征向量中提取原始3维参数"""
        return features[:, :3]

    def find_nearest_training_samples(self, target_features, k=3):
        """找到参数空间中最接近的训练样本"""
        target_params = self.extract_original_params(target_features.reshape(1, -1))[0]
        distances, indices = self.nn_searcher.kneighbors([target_params], n_neighbors=k)
        return indices[0], distances[0]

    def get_similar_training_modes(self, target_features, similarity_threshold=0.1):
        """获取相似参数对应的训练模态"""
        target_params = self.extract_original_params(target_features.reshape(1, -1))[0]
        indices, distances = self.find_nearest_training_samples(target_features)

        similar_modes_real = []
        similar_modes_imag = []
        similar_params = []

        for idx, dist in zip(indices, distances):
            if dist < similarity_threshold * np.max(self.training_params):
                similar_modes_real.append(self.training_modes_real[idx])
                similar_modes_imag.append(self.training_modes_imag[idx])
                similar_params.append(self.training_params[idx])

        if similar_modes_real:
            return (np.array(similar_modes_real),
                    np.array(similar_modes_imag),
                    np.array(similar_params))
        else:
            return (self.training_modes_real[indices[:3]],
                    self.training_modes_imag[indices[:3]],
                    self.training_params[indices[:3]])

    def apply_solution_space_constraint(self, pred_real, pred_imag, target_features, constraint_strength=0.5):
        """应用解空间约束到预测结果"""
        similar_real, similar_imag, similar_params = self.get_similar_training_modes(target_features)

        if len(similar_real) == 0:
            return pred_real, pred_imag

        similar_real_mean = np.mean(similar_real, axis=0)
        similar_real_std = np.std(similar_real, axis=0) + 1e-6
        similar_imag_mean = np.mean(similar_imag, axis=0)
        similar_imag_std = np.std(similar_imag, axis=0) + 1e-6

        real_zscore = np.abs(pred_real - similar_real_mean) / similar_real_std
        imag_zscore = np.abs(pred_imag - similar_imag_mean) / similar_imag_std

        real_weight = np.exp(-constraint_strength * real_zscore)
        imag_weight = np.exp(-constraint_strength * imag_zscore)

        constrained_real = real_weight * pred_real + (1 - real_weight) * similar_real_mean
        constrained_imag = imag_weight * pred_imag + (1 - imag_weight) * similar_imag_mean

        return constrained_real, constrained_imag

    def validate_prediction(self, pred_real, pred_imag, target_features, z_threshold=3.0):
        """验证预测结果是否在解空间合理范围内"""
        similar_real, similar_imag, _ = self.get_similar_training_modes(target_features)

        if len(similar_real) == 0:
            return True, "无相似训练样本可供验证"

        similar_mean = np.mean(similar_real, axis=0)
        similar_std = np.std(similar_real, axis=0) + 1e-6

        z_scores = np.abs(pred_real - similar_mean) / similar_std
        outlier_mask = z_scores > z_threshold
        outlier_ratio = np.sum(outlier_mask) / pred_real.size

        if outlier_ratio > 0.05:
            return False, f"预测异常点比例过高: {outlier_ratio:.2%}"
        else:
            return True, f"预测在解空间合理范围内 (异常点: {outlier_ratio:.2%})"


# ======================== 解空间约束网络（修复序列化问题） ========================
@tf.keras.utils.register_keras_serializable()
class SolutionSpaceConstrainedNN(Model):
    """基于解空间约束的神经网络 - 修复序列化问题"""

    def __init__(self, input_dim, output_dim, n_x, n_y, solution_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_x = n_x
        self.n_y = n_y
        self.solution_constraint = solution_constraint

        # 网络结构
        self.input_norm = BatchNormalization()
        self.dense_layers = [
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(256, activation='relu', kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dense(output_dim, activation='linear')
        ]

        # 约束强度参数
        self.constraint_strength = self.add_weight(
            name='constraint_strength',
            shape=(),
            initializer=tf.constant_initializer(0.5),
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.1, max_value=1.0),
            trainable=True
        )

    def get_config(self):
        """获取模型配置 - 修复序列化问题"""
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'n_x': self.n_x,
            'n_y': self.n_y,
            'solution_constraint': None,  # 不序列化solution_constraint
        })
        return config

    @classmethod
    def from_config(cls, config):
        """从配置创建模型实例 - 修复反序列化问题"""
        # 移除solution_constraint，因为它不能序列化
        config.pop('solution_constraint', None)
        return cls(**config)

    def call(self, inputs, training=None):
        # 前向传播
        x = self.input_norm(inputs)
        for layer in self.dense_layers:
            x = layer(x)

        # 如果是推理阶段，应用解空间约束
        if not training and self.solution_constraint is not None:
            batch_size = tf.shape(inputs)[0]
            constrained_outputs = []

            for i in range(batch_size):
                single_pred = x[i]
                single_features = inputs[i]

                pred_reshaped = tf.reshape(single_pred, [self.n_x, self.n_y, 2])
                pred_real = pred_reshaped[..., 0]
                pred_imag = pred_reshaped[..., 1]

                constrained_real, constrained_imag = tf.py_function(
                    self._apply_constraint_single,
                    [pred_real, pred_imag, single_features, self.constraint_strength],
                    [tf.float32, tf.float32]
                )

                constrained_real.set_shape([self.n_x, self.n_y])
                constrained_imag.set_shape([self.n_x, self.n_y])

                constrained_pred = tf.stack([constrained_real, constrained_imag], axis=-1)
                constrained_flat = tf.reshape(constrained_pred, [self.output_dim])
                constrained_outputs.append(constrained_flat)

            x = tf.stack(constrained_outputs, axis=0)

        return x

    def _apply_constraint_single(self, pred_real, pred_imag, features, strength):
        """应用约束到单个样本"""
        pred_real_np = pred_real.numpy()
        pred_imag_np = pred_imag.numpy()
        features_np = features.numpy()
        strength_np = strength.numpy()

        if self.solution_constraint is not None:
            constrained_real, constrained_imag = self.solution_constraint.apply_solution_space_constraint(
                pred_real_np, pred_imag_np, features_np, strength_np
            )
        else:
            constrained_real, constrained_imag = pred_real_np, pred_imag_np

        return constrained_real.astype(np.float32), constrained_imag.astype(np.float32)


# ======================== 改进的训练器 ========================
class SolutionSpaceTrainer:
    """基于解空间约束的训练器"""

    def __init__(self):
        self.best_loss = float('inf')
        self.best_model = None
        self.loss_history = []
        self.constraint_strength_history = []
        self.solution_constraint = None

    def prepare_data(self, params, modes_real, modes_imag):
        """准备训练数据"""
        print("🔧 准备训练数据...")

        n_samples = len(params)
        valid_mask = ~(np.isnan(params).any(axis=1) |
                       np.isnan(modes_real).any(axis=(1, 2)) |
                       np.isnan(modes_imag).any(axis=(1, 2)))

        params_clean = params[valid_mask]
        modes_real_clean = modes_real[valid_mask]
        modes_imag_clean = modes_imag[valid_mask]

        print(f"📊 有效样本数: {len(params_clean)}/{n_samples}")

        # 初始化解空间约束
        self.solution_constraint = SolutionSpaceConstraint(
            params_clean, modes_real_clean, modes_imag_clean
        )

        # 特征工程
        X = self._create_features(params_clean)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        # 准备输出数据
        y_flattened = []
        for i in range(len(modes_real_clean)):
            combined = np.stack([modes_real_clean[i], modes_imag_clean[i]], axis=-1)
            flattened = combined.reshape(-1)
            y_flattened.append(flattened)

        y = np.array(y_flattened, dtype=np.float32)

        return X_scaled, y, modes_real_clean[0].shape

    def _create_features(self, params):
        """创建特征"""
        original_params = params[:, :3]
        qin = original_params[:, 0]
        qout = original_params[:, 1]
        cin = original_params[:, 2]

        features = [
            original_params,
            (qin / (qout + 1e-6)).reshape(-1, 1),
            (cin / (qin + 1e-6)).reshape(-1, 1),
            (qin * cin).reshape(-1, 1),
            (qin ** 2).reshape(-1, 1),
            (qout ** 2).reshape(-1, 1),
            (cin ** 2).reshape(-1, 1)
        ]

        return np.hstack(features).astype(np.float32)

    def _safe_apply_gradients(self, grads, variables, optimizer):
        """安全地应用梯度"""
        valid_grads = []
        valid_vars = []

        for g, v in zip(grads, variables):
            if g is not None and g.shape == v.shape:
                valid_grads.append(g)
                valid_vars.append(v)

        if valid_grads:
            clipped_grads, _ = tf.clip_by_global_norm(valid_grads, 1.0)
            optimizer.apply_gradients(zip(clipped_grads, valid_vars))
            return True
        return False

    def train(self, params, modes_real, modes_imag, n_x, n_y, epochs=200):
        """训练模型"""
        X, y, output_shape = self.prepare_data(params, modes_real, modes_imag)

        if len(X) == 0:
            print("❌ 无有效训练数据")
            return None

        input_dim = X.shape[1]
        output_dim = y.shape[1]

        print(f"📐 输入维度: {input_dim}, 输出维度: {output_dim}")

        # 创建模型
        model = SolutionSpaceConstrainedNN(
            input_dim=input_dim,
            output_dim=output_dim,
            n_x=n_x,
            n_y=n_y,
            solution_constraint=self.solution_constraint
        )

        # 编译模型
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(optimizer=optimizer, loss='mse')

        # 训练验证分割
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"🔀 训练集: {X_train.shape}, 验证集: {X_val.shape}")

        # 训练循环
        best_weights = None
        patience = 20
        wait = 0

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predictions = model(X_train, training=True)
                loss = tf.reduce_mean(tf.square(y_train - predictions))

            grads = tape.gradient(loss, model.trainable_variables)
            self._safe_apply_gradients(grads, model.trainable_variables, optimizer)

            # 验证
            val_predictions = model(X_val, training=False)
            val_loss = tf.reduce_mean(tf.square(y_val - val_predictions))

            self.loss_history.append(loss.numpy())
            self.constraint_strength_history.append(model.constraint_strength.numpy())

            # 早停
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_weights = model.get_weights()
                wait = 0
                print(f"⭐ Epoch {epoch}: 最佳验证损失: {val_loss:.4f}")
            else:
                wait += 1

            if epoch % 10 == 0:
                print(f"⏳ Epoch {epoch}: loss={loss:.4f}, val_loss={val_loss:.4f}, "
                      f"约束强度: {model.constraint_strength.numpy():.3f}")

            if wait >= patience:
                print(f"⚠️ 早停触发于 epoch {epoch}")
                break

        if best_weights is not None:
            model.set_weights(best_weights)

        self.best_model = model
        self.output_shape = (n_x, n_y, 2)

        # 保存预处理工具
        self._save_preprocess_tools()

        return model

    def _save_preprocess_tools(self):
        """保存预处理工具"""
        tools_path = os.path.join(results_path, 'preprocess_tools')
        os.makedirs(tools_path, exist_ok=True)

        joblib.dump(self.scaler, os.path.join(tools_path, 'scaler.joblib'))
        joblib.dump(self.solution_constraint, os.path.join(tools_path, 'solution_constraint.joblib'))
        joblib.dump({'n_x': self.output_shape[0], 'n_y': self.output_shape[1]},
                    os.path.join(tools_path, 'output_info.joblib'))

        print(f"💾 预处理工具已保存至: {tools_path}")

    def save_training_plot(self):
        """保存训练过程图表"""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.constraint_strength_history)
        plt.xlabel('Epoch')
        plt.ylabel('Constraint Strength')
        plt.title('Constraint Strength Evolution')
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(results_path, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 训练图表已保存至: {plot_path}")


# ======================== 解空间预测器 ========================
class SolutionSpacePredictor:
    """基于解空间约束的预测器"""

    def __init__(self, model_path, preprocess_dir):
        # 加载模型和预处理工具
        custom_objects = {
            'SolutionSpaceConstrainedNN': SolutionSpaceConstrainedNN,
            'SolutionSpaceConstraint': SolutionSpaceConstraint
        }

        with tf.keras.utils.custom_object_scope(custom_objects):
            self.model = load_model(model_path)

        self.scaler = joblib.load(os.path.join(preprocess_dir, 'scaler.joblib'))
        self.solution_constraint = joblib.load(os.path.join(preprocess_dir, 'solution_constraint.joblib'))
        output_info = joblib.load(os.path.join(preprocess_dir, 'output_info.joblib'))
        self.n_x = output_info['n_x']
        self.n_y = output_info['n_y']

        print(f"✅ 解空间预测器加载完成")
        print(f"📐 输出形状: {self.n_x}x{self.n_y}")

    def create_features(self, params):
        """创建特征"""
        original_params = params[:, :3]
        qin = original_params[:, 0]
        qout = original_params[:, 1]
        cin = original_params[:, 2]

        features = [
            original_params,
            (qin / (qout + 1e-6)).reshape(-1, 1),
            (cin / (qin + 1e-6)).reshape(-1, 1),
            (qin * cin).reshape(-1, 1),
            (qin ** 2).reshape(-1, 1),
            (qout ** 2).reshape(-1, 1),
            (cin ** 2).reshape(-1, 1)
        ]

        features_array = np.hstack(features).astype(np.float32)
        return self.scaler.transform(features_array)

    def save_prediction_results(self, pred_real, pred_imag, pred_complex, qin, qout, cin, is_valid, validation_msg):
        """保存预测结果为npy文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"prediction_qin{qin}_qout{qout}_cin{cin}_{timestamp}"

        # 保存实部
        real_path = os.path.join(results_path, f"{prefix}_real.npy")
        np.save(real_path, pred_real)

        # 保存虚部
        imag_path = os.path.join(results_path, f"{prefix}_imag.npy")
        np.save(imag_path, pred_imag)

        # 保存复数形式
        complex_path = os.path.join(results_path, f"{prefix}_complex.npy")
        np.save(complex_path, pred_complex)

        # 保存预测信息
        info_path = os.path.join(results_path, f"{prefix}_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"预测参数: Qin={qin}, Qout={qout}, Cin={cin}\n")
            f.write(f"时间戳: {timestamp}\n")
            f.write(f"验证状态: {'通过' if is_valid else '警告'}\n")
            f.write(f"验证信息: {validation_msg}\n")
            f.write(f"实部形状: {pred_real.shape}\n")
            f.write(f"虚部形状: {pred_imag.shape}\n")

        print(f"💾 预测结果已保存:")
        print(f"  实部: {real_path}")
        print(f"  虚部: {imag_path}")
        print(f"  复数: {complex_path}")
        print(f"  信息: {info_path}")

        return real_path, imag_path, complex_path, info_path

    def predict(self, qin, qout, cin, apply_constraint=True, save_results=True):
        """执行预测"""
        params = np.array([[qin, qout, cin]], dtype=np.float32)
        features = self.create_features(params)

        predictions = self.model.predict(features, verbose=0)

        pred_flattened = predictions[0]
        pred_reshaped = pred_flattened.reshape(self.n_x, self.n_y, 2)

        pred_real = pred_reshaped[..., 0]
        pred_imag = pred_reshaped[..., 1]

        # 应用解空间约束
        if apply_constraint and self.solution_constraint is not None:
            target_params = np.array([qin, qout, cin])
            similar_indices, distances = self.solution_constraint.find_nearest_training_samples(features[0])

            print(f"🔍 最近训练样本:")
            for i, idx in enumerate(similar_indices[:3]):
                sim_params = self.solution_constraint.training_params[idx]
                print(
                    f"  样本 {i + 1}: Qin={sim_params[0]}, Qout={sim_params[1]}, Cin={sim_params[2]} (距离: {distances[i]:.2f})")

            constrained_real, constrained_imag = self.solution_constraint.apply_solution_space_constraint(
                pred_real, pred_imag, features[0]
            )
            pred_real, pred_imag = constrained_real, constrained_imag

        pred_complex = pred_real + 1j * pred_imag

        # 验证
        if self.solution_constraint is not None:
            is_valid, validation_msg = self.solution_constraint.validate_prediction(
                pred_real, pred_imag, features[0]
            )
        else:
            is_valid, validation_msg = True, "无约束信息可供验证"

        # 保存结果
        saved_paths = None
        if save_results:
            saved_paths = self.save_prediction_results(
                pred_real, pred_imag, pred_complex, qin, qout, cin, is_valid, validation_msg
            )

        return pred_real, pred_imag, pred_complex, is_valid, validation_msg, saved_paths


# ======================== 数据加载函数 ========================
def load_all_samples():
    """加载所有样本数据"""
    samples = []
    sample_params = []
    qin_values = [60, 90, 120, 150, 180]
    qout_values = [30, 60, 90, 120, 150, 180]
    cin_values = [10, 40, 70, 100, 130, 160]

    print("=" * 60)
    print("🔍 加载样本数据...")
    print("=" * 60)

    total_samples = valid_samples = 0

    for qin, qout, cin in product(qin_values, qout_values, cin_values):
        total_samples += 1
        modes_file = f"modes_{qin}{qout}_{qin}{qout}-{cin}.npy"
        alt_modes_file = f"modes_{str(qin).zfill(3)}{str(qout).zfill(3)}_{str(qin).zfill(3)}{str(qout).zfill(3)}-{cin}.npy"

        try:
            modes_path = os.path.join(folder_path, modes_file)
            if not os.path.exists(modes_path):
                modes_path = os.path.join(folder_path, alt_modes_file)
                if not os.path.exists(modes_path):
                    continue

            modes = np.load(modes_path)

            if np.iscomplexobj(modes):
                modes_real, modes_imag = np.real(modes), np.imag(modes)
            else:
                modes_real, modes_imag = modes, np.zeros_like(modes)

            modes_real = np.squeeze(modes_real)
            modes_imag = np.squeeze(modes_imag)

            if np.isnan(modes_real).any() or np.isnan(modes_imag).any():
                continue

            samples.append({
                'modes_real': modes_real,
                'modes_imag': modes_imag,
                'params': [qin, qout, cin]
            })
            sample_params.append([qin, qout, cin])
            valid_samples += 1

            if valid_samples % 10 == 0:
                print(f"✅ 已加载 {valid_samples} 个样本...")

        except Exception as e:
            continue

    if not samples:
        return np.array([]), np.array([]), np.array([])

    params_array = np.array([s['params'] for s in samples])
    modes_real_array = np.array([s['modes_real'] for s in samples])
    modes_imag_array = np.array([s['modes_imag'] for s in samples])

    print("=" * 60)
    print(f"📊 有效样本: {valid_samples}/{total_samples}")
    print("=" * 60)

    return params_array, modes_real_array, modes_imag_array


# ======================== 主程序 ========================
if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "=" * 60)
    print("🚀 SPOD模态预测 - 解空间约束版（修复序列化问题）")
    print("=" * 60)

    # 1. 加载数据
    params, modes_real, modes_imag = load_all_samples()

    if len(params) == 0:
        print("❌ 无有效数据，程序终止!")
        exit(1)

    n_samples, n_x, n_y = modes_real.shape
    print(f"📐 数据维度: {n_samples} 个样本, 空间 {n_x}x{n_y}")

    # 2. 训练模型
    trainer = SolutionSpaceTrainer()
    model = trainer.train(params, modes_real, modes_imag, n_x, n_y, epochs=150)

    if model is None:
        print("❌ 训练失败")
        exit(1)

    # 3. 保存模型
    model_path = os.path.join(results_path, "solution_space_model.keras")
    model.save(model_path)
    print(f"💾 模型已保存: {model_path}")

    # 保存训练图表
    trainer.save_training_plot()

    # 4. 测试预测
    print("\n🔮 开始解空间约束预测测试...")
    predictor = SolutionSpacePredictor(model_path, os.path.join(results_path, 'preprocess_tools'))

    # 测试用例
    test_cases = [
        (60, 60, 10),  # 已知样本
        (60, 60, 11),  # 接近已知样本
        (120, 90, 70),  # 已知样本
        (120, 90, 75),  # 接近已知样本
        (180, 150, 130)  # 已知样本
    ]

    for qin, qout, cin in test_cases:
        print(f"\n" + "=" * 50)
        print(f"🎯 预测参数: Qin={qin}, Qout={qout}, Cin={cin}")

        # 执行预测
        pred_real, pred_imag, pred_complex, is_valid, validation_msg, saved_paths = predictor.predict(
            qin, qout, cin, apply_constraint=True, save_results=True
        )

        print(f"🔍 {validation_msg}")
        print(f"✅ 验证状态: {'通过' if is_valid else '警告'}")

    # 5. 性能统计
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"⏱️ 总耗时: {total_time:.2f}秒")
    print(f"📈 最终训练损失: {trainer.loss_history[-1]:.4f}")
    print("=" * 60)