import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support, f1_score
import seaborn as sns
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train.data_loader import load_processed_dataset, prepare_data_for_cnn

# Định nghĩa lại môi trường để tương thích với môi trường trong train.py
class SleepEnvironment(gym.Env):
    """Môi trường Gym cho phân loại giấc ngủ để đánh giá"""
    
    def __init__(self, data_loader, device='cuda'):
        super(SleepEnvironment, self).__init__()
        
        self.data_loader = data_loader
        self.device = device
        
        # Khởi tạo biến theo dõi state
        self.current_batch = None
        self.current_idx = 0
        self.data_iter = iter(data_loader)
        self.get_new_batch()
        
        # Lưu lại toàn bộ nhãn và dự đoán
        self.all_predictions = []
        self.all_true_labels = []
        
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
    
    def reset(self):
        """Reset môi trường và trả về state đầu tiên"""
        self.get_new_batch()
        return self.current_batch[0][self.current_idx].numpy()
    
def step(self, action):
    """Thực hiện hành động và trả về state mới, reward, done, truncated, info"""
    # Lấy nhãn thật
    true_label = self.current_batch[1][self.current_idx].item()
    
    # Tính toán reward: +1 đúng, -1 sai
    reward = 1.0 if action == true_label else -1.0
    
    # Cập nhật index và kiểm tra done
    self.current_idx += 1
    done = self.current_idx >= len(self.current_batch[0])
    truncated = False  # Thêm trường truncated mới trong gymnasium
    
    # Nếu đã hết batch hiện tại, lấy batch mới
    if done:
        self.get_new_batch()
    
    # Lấy state mới
    next_state = self.current_batch[0][self.current_idx].numpy()
    
    # Info dictionary cho debugging
    info = {'true_label': true_label}
    
    return next_state, reward, done, truncated, info
    
def make_env(data_loader):
    """Helper function để tạo môi trường"""
    def _init():
        env = SleepEnvironment(data_loader)
        return env
    return _init

def evaluate_sb3_model(model, test_loader):
    """Đánh giá mô hình SB3 trên tập test"""
    
    # Tạo môi trường đánh giá
    env = SleepEnvironment(test_loader)
    
    # Thực hiện đánh giá
    done = False
    obs = env.reset()
    
    while len(env.all_predictions) < len(test_loader.dataset):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    return np.array(env.all_predictions), np.array(env.all_true_labels)

def evaluate_and_display_metrics(predictions, true_labels, class_names):
    """
    Tính toán và hiển thị các metrics phổ biến
    """
    # Accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Precision, Recall, F1-score cho từng lớp
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, average=None, labels=range(len(class_names))
    )
    
    # F1-score tổng thể
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    
    print(f"F1-score (macro): {f1_macro:.4f}")
    print(f"F1-score (weighted): {f1_weighted:.4f}")
    
    # Tạo DataFrame để hiển thị kết quả trực quan
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Support': support
    })
    
    print("\nMetrics cho từng giai đoạn giấc ngủ:")
    print(metrics_df.to_string(index=False))
    
    # Vẽ biểu đồ
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Precision
    axs[0].bar(class_names, precision)
    axs[0].set_title('Precision')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlabel('Giai đoạn giấc ngủ')
    
    # Recall
    axs[1].bar(class_names, recall)
    axs[1].set_title('Recall')
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('Giai đoạn giấc ngủ')
    
    # F1-score
    axs[2].bar(class_names, f1)
    axs[2].set_title('F1-score')
    axs[2].set_ylim(0, 1)
    axs[2].set_xlabel('Giai đoạn giấc ngủ')
    
    plt.tight_layout()
    plt.savefig('sb3_metrics.png')
    plt.show()
    
    return metrics_df

# Main evaluation function
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Nạp dữ liệu test
    processed_dir = "D:/codeN/Biomedical/Paper/RL-CNN_Resnet50/data"
    _, _, _, _, X_test, y_test = load_processed_dataset(
        processed_dir=processed_dir,
        subjects=None,
        test_size=0.2,
        val_size=0.2
    )
    
    X_test_tensor, y_test_tensor = prepare_data_for_cnn(X_test, y_test)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Nạp model SB3
    model = A2C.load("sb3_a2c_sleep_model")
    
    # Đánh giá
    predictions, true_labels = evaluate_sb3_model(model, test_loader)
    
    # Tên các giai đoạn giấc ngủ
    stage_names = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
    
    # Tính toán và hiển thị các metrics
    metrics_df = evaluate_and_display_metrics(predictions, true_labels, stage_names)
    
    # Báo cáo phân loại đầy đủ
    print("\nBáo cáo phân loại chi tiết:")
    report = classification_report(true_labels, predictions, target_names=stage_names, digits=4)
    print(report)
    
    # Ma trận nhầm lẫn
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=stage_names, yticklabels=stage_names)
    plt.title('Confusion Matrix - SB3 A2C Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('sb3_confusion_matrix.png')
    plt.show()
    
    # So sánh với baseline (nếu có)
    try:
        # Load kết quả baseline (mô hình không dùng RL)
        baseline_results = np.load('baseline_results.npz')
        baseline_pred = baseline_results['predictions']
        baseline_true = baseline_results['true_labels']
        
        # Tính F1-score
        sb3_f1 = f1_score(true_labels, predictions, average='weighted')
        baseline_f1 = f1_score(baseline_true, baseline_pred, average='weighted')
        
        # So sánh
        print("\nSo sánh với baseline:")
        print(f"F1-score (SB3): {sb3_f1:.4f}")
        print(f"F1-score (Baseline): {baseline_f1:.4f}")
        print(f"Cải thiện: {(sb3_f1 - baseline_f1) / baseline_f1 * 100:.2f}%")
    except:
        print("\nKhông tìm thấy kết quả baseline để so sánh") 