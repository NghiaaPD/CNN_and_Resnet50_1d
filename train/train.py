import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import load_processed_dataset, prepare_data_for_cnn
from models.CNN_Resnet50_RL import CNN_Resnet50_Features

# Tích hợp CNN_Resnet50_Features với SB3
class CustomCNN(BaseFeaturesExtractor):
    """
    Tích hợp CNN_Resnet50_Features với SB3
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        
        # Cấu trúc của observation space
        n_channels = observation_space.shape[0]
        
        # Sử dụng CNN_Resnet50_Features làm feature extractor
        self.feature_extractor = CNN_Resnet50_Features(
            num_channels=n_channels, 
            num_classes=5
        )
        
        # Linear layer để map đến số chiều đặc trưng cần thiết
        self.linear = nn.Linear(512 * 4, features_dim)  # Giả định ResNet50 có 512*4 đặc trưng
        
    def forward(self, observations):
        # Trích xuất đặc trưng với CNN_Resnet50_Features
        features = self.feature_extractor(observations)
        
        # Map về số chiều đặc trưng yêu cầu
        x = self.linear(features)
        return x

class SleepEnvironment(gym.Env):
    """Môi trường Gym cho phân loại giấc ngủ"""
    
    def __init__(self, data_loader, device='cuda'):
        super(SleepEnvironment, self).__init__()
        
        self.data_loader = data_loader
        self.device = device
        
        # Khởi tạo biến theo dõi state
        self.current_batch = None
        self.current_idx = 0
        self.data_iter = iter(data_loader)
        self.get_new_batch()
        
        # Định nghĩa không gian action & observation
        for inputs, labels in data_loader:
            sample_shape = inputs[0].numpy().shape
            self.num_classes = len(torch.unique(labels))
            break
            
        self.action_space = spaces.Discrete(self.num_classes)  # 5 giai đoạn giấc ngủ
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=sample_shape,
            dtype=np.float32
        )
        
    def get_new_batch(self):
        try:
            self.current_batch = next(self.data_iter)
            self.current_idx = 0
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            self.current_batch = next(self.data_iter)
            self.current_idx = 0
    
    def reset(self, seed=None, options=None):
        """Reset môi trường và trả về state đầu tiên"""
        # Thiết lập seed nếu được cung cấp
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.get_new_batch()
        obs = self.current_batch[0][self.current_idx].numpy()
        info = {}  # Dictionary thông tin bổ sung
        return obs, info  # Trả về tuple (observation, info)
    
    def step(self, action):
        """Thực hiện hành động và trả về state mới, reward, done, truncated, info"""
        # Lấy nhãn thật
        true_label = self.current_batch[1][self.current_idx].item()
        
        # Tính toán reward: +1 đúng, -1 sai
        reward = 1.0 if action == true_label else -1.0
        
        # Cập nhật index và kiểm tra done
        self.current_idx += 1
        terminated = self.current_idx >= len(self.current_batch[0])
        truncated = False  # Thêm trường truncated mới trong gymnasium
        
        # Nếu đã hết batch hiện tại, lấy batch mới
        if terminated:
            self.get_new_batch()
        
        # Lấy state mới
        next_state = self.current_batch[0][self.current_idx].numpy()
        
        # Info dictionary cho debugging
        info = {'true_label': true_label}
        
        return next_state, reward, terminated, truncated, info

def make_env(data_loader, rank=0, seed=0):
    """Helper function để tạo môi trường"""
    def _init():
        env = SleepEnvironment(data_loader)
        env = Monitor(env, f"./logs/sleep_env_{rank}")
        env.reset(seed=seed + rank)  # Thiết lập seed khi reset
        return env
    return _init

# Main training function
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tạo thư mục logs
    os.makedirs("logs", exist_ok=True)
    
    # Tham số
    batch_size = 32
    learning_rate = 0.0001  # Giảm học tập để ổn định hơn
    total_timesteps = 100000  # Số bước huấn luyện
    num_classes = 5
    n_steps = 8  # Tăng từ 5 lên 8
    gamma = 0.99  # Giữ nguyên discount factor
    ent_coef = 0.01  # Thêm hệ số entropy để tăng exploration
    
    # Nạp dữ liệu
    processed_dir = "D:/codeN/Biomedical/Paper/RL-CNN_Resnet50/data"
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dataset(
        processed_dir=processed_dir,
        subjects=None,
        test_size=0.2,
        val_size=0.2
    )
    
    # Chuẩn bị dữ liệu
    X_train_tensor, y_train_tensor = prepare_data_for_cnn(X_train, y_train)
    X_val_tensor, y_val_tensor = prepare_data_for_cnn(X_val, y_val)
    X_test_tensor, y_test_tensor = prepare_data_for_cnn(X_test, y_test)
    
    # Tạo DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Tạo môi trường RL
    env = DummyVecEnv([make_env(train_loader, rank=0)])
    
    # Tạo eval environment cho callback
    eval_env = DummyVecEnv([make_env(val_loader, rank=0)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=500,
        deterministic=True,
        render=False
    )
    
    # Định nghĩa policy_kwargs để sử dụng CNN_Resnet50_Features làm feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=1024),  # Tăng từ 512 lên 1024
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]  # Tăng kích thước mạng
    )
    
    # Khởi tạo A2C với custom feature extractor
    model = A2C(
        policy="CnnPolicy",  # Sử dụng policy có sẵn của SB3 với CNN
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,  # Số bước cho mỗi update
        gamma=gamma,  # Discount factor
        policy_kwargs=policy_kwargs,
        verbose=1
    )
    
    # Huấn luyện agent
    print("Bắt đầu huấn luyện RL agent với Stable-Baselines3...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Lưu model
    model.save("model")
    
    # Đánh giá mô hình cuối cùng
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Tạo môi trường test để hiển thị kết quả
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    test_env = DummyVecEnv([make_env(test_loader, rank=0)])
    
    # Hiển thị kết quả trên tập test
    mean_test_reward, _ = evaluate_policy(
        model, test_env, n_eval_episodes=10, deterministic=True
    )
    print(f"Test reward: {mean_test_reward:.2f}")
    
    print("Huấn luyện hoàn thành. Đánh giá chi tiết có thể thực hiện với evaluate_rl.py") 