import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import matplotlib.ticker as ticker

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import load_processed_dataset, prepare_data_for_cnn
from models.CNN_Resnet50 import CNN_Resnet50

def evaluate_model_metrics(model, test_loader, criterion, device='cuda', class_names=None):
    """
    Đánh giá mô hình và tính toán các metrics
    
    Parameters:
    -----------
    model: torch.nn.Module
        Mô hình cần đánh giá
    test_loader: torch.utils.data.DataLoader
        DataLoader cho dữ liệu test
    criterion: torch.nn.modules.loss
        Hàm mất mát
    device: str
        Thiết bị tính toán ('cuda' hoặc 'cpu')
    class_names: list
        Tên các lớp
        
    Returns:
    --------
    metrics: dict
        Dictionary chứa các metrics
    """
    if class_names is None:
        class_names = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
    
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # Lưu xác suất cho ROC curve
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Đánh giá"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Lấy xác suất thông qua softmax
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # Thống kê
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            # Lưu dự đoán và nhãn thật
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Chuyển sang numpy array
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Tính các metrics
    test_loss = test_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Tính các metrics cho từng lớp và trung bình
    precision_micro = precision_score(all_labels, all_preds, average='micro')
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    precision_per_class = precision_score(all_labels, all_preds, average=None)
    
    recall_micro = recall_score(all_labels, all_preds, average='micro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    recall_per_class = recall_score(all_labels, all_preds, average=None)
    
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    
    # Tạo confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Tạo dictionary chứa tất cả metrics
    metrics = {
        'loss': test_loss,
        'accuracy': accuracy,
        'precision': {
            'micro': precision_micro,
            'macro': precision_macro,
            'weighted': precision_weighted,
            'per_class': {class_names[i]: precision_per_class[i] for i in range(len(class_names))}
        },
        'recall': {
            'micro': recall_micro,
            'macro': recall_macro,
            'weighted': recall_weighted,
            'per_class': {class_names[i]: recall_per_class[i] for i in range(len(class_names))}
        },
        'f1': {
            'micro': f1_micro,
            'macro': f1_macro,
            'weighted': f1_weighted,
            'per_class': {class_names[i]: f1_per_class[i] for i in range(len(class_names))}
        },
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels,
        'probabilities': all_probs,
        'class_names': class_names
    }
    
    return metrics

def print_metrics_report(metrics):
    """
    In báo cáo chi tiết về các metrics
    
    Parameters:
    -----------
    metrics: dict
        Dictionary chứa các metrics
    """
    class_names = metrics['class_names']
    
    print("\n" + "="*50)
    print("BÁO CÁO ĐÁNH GIÁ MÔ HÌNH")
    print("="*50)
    
    print(f"\nLoss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print("\nPrecision:")
    print(f"  Micro-average: {metrics['precision']['micro']:.4f}")
    print(f"  Macro-average: {metrics['precision']['macro']:.4f}")
    print(f"  Weighted-average: {metrics['precision']['weighted']:.4f}")
    print("  Theo từng lớp:")
    for class_name, value in metrics['precision']['per_class'].items():
        print(f"    {class_name}: {value:.4f}")
    
    print("\nRecall:")
    print(f"  Micro-average: {metrics['recall']['micro']:.4f}")
    print(f"  Macro-average: {metrics['recall']['macro']:.4f}")
    print(f"  Weighted-average: {metrics['recall']['weighted']:.4f}")
    print("  Theo từng lớp:")
    for class_name, value in metrics['recall']['per_class'].items():
        print(f"    {class_name}: {value:.4f}")
    
    print("\nF1 Score:")
    print(f"  Micro-average: {metrics['f1']['micro']:.4f}")
    print(f"  Macro-average: {metrics['f1']['macro']:.4f}")
    print(f"  Weighted-average: {metrics['f1']['weighted']:.4f}")
    print("  Theo từng lớp:")
    for class_name, value in metrics['f1']['per_class'].items():
        print(f"    {class_name}: {value:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(
        metrics['true_labels'], 
        metrics['predictions'], 
        target_names=class_names, 
        digits=4
    )
    print(report)
    
    print("="*50)

def plot_metrics_visualizations(metrics, save_dir=None):
    """
    Trực quan hóa các metrics
    
    Parameters:
    -----------
    metrics: dict
        Dictionary chứa các metrics
    save_dir: str
        Thư mục lưu hình ảnh
    """
    class_names = metrics['class_names']
    cm = metrics['confusion_matrix']
    
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Confusion Matrix heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)', fontsize=16)
    plt.xlabel('Dự đoán', fontsize=14)
    plt.ylabel('Giá trị thực', fontsize=14)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Precision, Recall, F1 per class
    plt.figure(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.25
    
    precision_values = [metrics['precision']['per_class'][c] for c in class_names]
    recall_values = [metrics['recall']['per_class'][c] for c in class_names]
    f1_values = [metrics['f1']['per_class'][c] for c in class_names]
    
    plt.bar(x - width, precision_values, width, label='Precision')
    plt.bar(x, recall_values, width, label='Recall')
    plt.bar(x + width, f1_values, width, label='F1 Score')
    
    plt.axhline(y=metrics['precision']['macro'], color='r', linestyle='--', alpha=0.3)
    plt.axhline(y=metrics['recall']['macro'], color='g', linestyle='--', alpha=0.3)
    plt.axhline(y=metrics['f1']['macro'], color='b', linestyle='--', alpha=0.3)
    
    plt.xlabel('Lớp')
    plt.ylabel('Giá trị')
    plt.title('Precision, Recall và F1 Score theo từng lớp')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics_per_class.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Normalized Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Xử lý division by zero
    cm_normalized = np.nan_to_num(cm_normalized)
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    plt.title('Ma trận nhầm lẫn đã chuẩn hóa', fontsize=16)
    plt.xlabel('Dự đoán', fontsize=14)
    plt.ylabel('Giá trị thực', fontsize=14)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Tạo bảng so sánh metrics
    fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
    
    metrics_data = {
        'Lớp': class_names,
        'Precision': precision_values,
        'Recall': recall_values,
        'F1 Score': f1_values
    }
    
    # Thêm hàng cho trung bình
    metrics_data['Lớp'].append('Macro Avg')
    metrics_data['Precision'].append(metrics['precision']['macro'])
    metrics_data['Recall'].append(metrics['recall']['macro'])
    metrics_data['F1 Score'].append(metrics['f1']['macro'])
    
    metrics_data['Lớp'].append('Weighted Avg')
    metrics_data['Precision'].append(metrics['precision']['weighted'])
    metrics_data['Recall'].append(metrics['recall']['weighted'])
    metrics_data['F1 Score'].append(metrics['f1']['weighted'])
    
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=[[metrics_data['Lớp'][i],
                               f"{metrics_data['Precision'][i]:.4f}",
                               f"{metrics_data['Recall'][i]:.4f}",
                               f"{metrics_data['F1 Score'][i]:.4f}"] 
                              for i in range(len(metrics_data['Lớp']))],
                    colLabels=['Lớp', 'Precision', 'Recall', 'F1 Score'],
                    loc='center', cellLoc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Bảng tổng hợp các chỉ số đánh giá', fontsize=16, pad=20)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'metrics_table.png'), dpi=300, bbox_inches='tight')
    plt.show()

def export_metrics_to_csv(metrics, filepath):
    """
    Xuất metrics ra file CSV
    
    Parameters:
    -----------
    metrics: dict
        Dictionary chứa các metrics
    filepath: str
        Đường dẫn file CSV
    """
    class_names = metrics['class_names']
    
    # Tạo data cho từng lớp
    data = []
    for i, class_name in enumerate(class_names):
        data.append({
            'Class': class_name,
            'Precision': metrics['precision']['per_class'][class_name],
            'Recall': metrics['recall']['per_class'][class_name],
            'F1 Score': metrics['f1']['per_class'][class_name]
        })
    
    # Thêm trung bình
    data.append({
        'Class': 'Macro Average',
        'Precision': metrics['precision']['macro'],
        'Recall': metrics['recall']['macro'],
        'F1 Score': metrics['f1']['macro']
    })
    
    data.append({
        'Class': 'Weighted Average',
        'Precision': metrics['precision']['weighted'],
        'Recall': metrics['recall']['weighted'],
        'F1 Score': metrics['f1']['weighted']
    })
    
    data.append({
        'Class': 'Overall Accuracy',
        'Precision': metrics['accuracy'],
        'Recall': metrics['accuracy'],
        'F1 Score': metrics['accuracy']
    })
    
    # Xuất ra CSV
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Đã xuất metrics ra file: {filepath}")

if __name__ == "__main__":
    # Cấu hình thiết bị tính toán
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")
    
    # Tham số
    batch_size = 64
    num_classes = 5  # 5 lớp: Wake, Stage 1, Stage 2, Stage 3, REM
    class_names = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
    
    # Đường dẫn dữ liệu
    processed_dir = "D:/codeN/Biomedical/CNN_and_Resnet50_1d/data"
    model_path = "best_model.pth"  # Đường dẫn đến model đã huấn luyện
    
    # Nạp dữ liệu
    print("Đang nạp dữ liệu...")
    _, _, _, _, X_test, y_test = load_processed_dataset(
        processed_dir=processed_dir,
        subjects=None,  # Sử dụng tất cả subjects
        test_size=0.2,
        val_size=0.2
    )
    
    # Chuyển dữ liệu sang tensor và chuẩn bị cho CNN
    X_test_tensor, y_test_tensor = prepare_data_for_cnn(X_test, y_test)
    
    # Tạo DataLoader
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Tạo model và nạp trọng số đã huấn luyện
    print("Đang nạp model...")
    model = CNN_Resnet50(num_classes=num_classes, num_channels=X_test.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Định nghĩa hàm mất mát
    criterion = nn.CrossEntropyLoss()
    
    # Đánh giá model và tính toán metrics
    print("Đang đánh giá model...")
    metrics = evaluate_model_metrics(model, test_loader, criterion, device, class_names)
    
    # In báo cáo
    print_metrics_report(metrics)
    
    # Trực quan hóa metrics
    output_dir = "evaluation_results"
    plot_metrics_visualizations(metrics, save_dir=output_dir)
    
    # Xuất metrics ra file CSV
    export_metrics_to_csv(metrics, os.path.join(output_dir, "metrics_report.csv"))
    
    print(f"Đã hoàn thành đánh giá model. Kết quả được lưu trong thư mục: {output_dir}") 