import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import time
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader import load_processed_dataset, prepare_data_for_cnn
from models.CNN_Resnet50 import CNN_Resnet50

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Huấn luyện mô hình
    
    Parameters:
    -----------
    model: torch.nn.Module
        Mô hình cần huấn luyện
    train_loader: torch.utils.data.DataLoader
        DataLoader cho dữ liệu huấn luyện
    val_loader: torch.utils.data.DataLoader
        DataLoader cho dữ liệu validation
    criterion: torch.nn.modules.loss
        Hàm mất mát
    optimizer: torch.optim
        Thuật toán tối ưu
    num_epochs: int
        Số epoch huấn luyện
    device: str
        Thiết bị tính toán ('cuda' hoặc 'cpu')
        
    Returns:
    --------
    history: dict
        Lịch sử huấn luyện (loss, accuracy)
    """
    model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print results
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} - {elapsed_time:.2f}s - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Đã lưu model tốt nhất với Val Acc: {val_acc:.4f}")
    
    return history

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    Đánh giá mô hình trên tập test
    
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
        
    Returns:
    --------
    test_loss: float
        Giá trị mất mát trên tập test
    test_acc: float
        Độ chính xác trên tập test
    all_preds: numpy.ndarray
        Tất cả các dự đoán
    all_labels: numpy.ndarray
        Tất cả các nhãn thật
    """
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / test_total
    
    return test_loss, test_acc, np.array(all_preds), np.array(all_labels)

def plot_training_history(history):
    """
    Vẽ đồ thị lịch sử huấn luyện
    
    Parameters:
    -----------
    history: dict
        Lịch sử huấn luyện (loss, accuracy)
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Vẽ ma trận nhầm lẫn
    
    Parameters:
    -----------
    true_labels: numpy.ndarray
        Nhãn thật
    predictions: numpy.ndarray
        Dự đoán của mô hình
    class_names: list
        Tên các lớp
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Cấu hình thiết bị tính toán
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Tham số cho huấn luyện
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 30
    num_classes = 5  # 5 lớp: Wake, Stage 1, Stage 2, Stage 3, REM
    
    # Đường dẫn dữ liệu
    processed_dir = "D:/codeN/Biomedical/Paper/RL-CNN_Resnet50/data"
    
    # Nạp dữ liệu
    print("Đang nạp dữ liệu...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dataset(
        processed_dir=processed_dir,
        subjects=None,  # Sử dụng tất cả subjects
        test_size=0.2,
        val_size=0.2
    )
    
    # Chuyển dữ liệu sang tensor và chuẩn bị cho CNN
    X_train_tensor, y_train_tensor = prepare_data_for_cnn(X_train, y_train)
    X_val_tensor, y_val_tensor = prepare_data_for_cnn(X_val, y_val)
    X_test_tensor, y_test_tensor = prepare_data_for_cnn(X_test, y_test)
    
    # Tạo DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Tạo model sử dụng CNN_Resnet50 đã định nghĩa sẵn
    print("Khởi tạo model CNN-ResNet50...")
    model = CNN_Resnet50(num_classes=num_classes, num_channels=X_train.shape[1])
    
    # Định nghĩa hàm mất mát và thuật toán tối ưu
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Huấn luyện model
    print("Bắt đầu huấn luyện model...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, 
                         num_epochs=num_epochs, device=device)
    
    # Vẽ đồ thị lịch sử huấn luyện
    plot_training_history(history)
    
    # Nạp model tốt nhất và đánh giá
    print("Nạp model tốt nhất và đánh giá trên tập test...")
    model.load_state_dict(torch.load('best_model.pth'))
    
    test_loss, test_acc, predictions, true_labels = evaluate_model(
        model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # In báo cáo phân loại
    stage_names = ["Wake", "Stage 1", "Stage 2", "Stage 3", "REM"]
    print("\nBáo cáo phân loại:")
    report = classification_report(true_labels, predictions, 
                                 target_names=stage_names, digits=4)
    print(report)
    
    # Vẽ ma trận nhầm lẫn
    plot_confusion_matrix(true_labels, predictions, stage_names)
