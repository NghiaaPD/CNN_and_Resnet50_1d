import os
import numpy as np
import glob
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch

def load_processed_dataset(processed_dir, subjects=None, test_size=0.2, val_size=0.2, random_state=42):
    """
    Tải dữ liệu đã được tiền xử lý
    
    Parameters:
    -----------
    processed_dir: str
        Đường dẫn đến thư mục chứa dữ liệu đã tiền xử lý
    subjects: int, optional
        Số lượng subjects cần tải
    test_size: float
        Tỷ lệ dữ liệu test
    val_size: float
        Tỷ lệ dữ liệu validation
    random_state: int
        Seed cho quá trình phân chia dữ liệu ngẫu nhiên
        
    Returns:
    --------
    X_train, y_train, X_val, y_val, X_test, y_test: numpy.ndarray
        Dữ liệu đã phân chia thành tập train, validation và test
    """
    # Tìm tất cả các file đã xử lý
    processed_files = []
    
    # Lấy danh sách thư mục subject
    subject_dirs = sorted(glob.glob(os.path.join(processed_dir, "sub-*")))
    
    if subjects:
        subject_dirs = subject_dirs[:subjects]
    
    for sub_dir in subject_dirs:
        # Tìm các file đã xử lý
        h5_files = glob.glob(os.path.join(sub_dir, "*_processed.h5"))
        npz_files = glob.glob(os.path.join(sub_dir, "*_processed.npz"))
        
        processed_files.extend(h5_files + npz_files)
    
    # Chia thành tập train, validation và test
    train_files, test_files = train_test_split(processed_files, test_size=test_size, random_state=random_state)
    train_files, val_files = train_test_split(train_files, test_size=val_size/(1-test_size), random_state=random_state)
    
    print(f"Số lượng file train: {len(train_files)}")
    print(f"Số lượng file validation: {len(val_files)}")
    print(f"Số lượng file test: {len(test_files)}")
    
    # Hàm đọc dữ liệu từ file HDF5 hoặc NPZ
    def load_file(file_path):
        if file_path.endswith('.h5'):
            with h5py.File(file_path, 'r') as f:
                X = f['data'][:]
                y = f['labels'][:]
                return X, y
        elif file_path.endswith('.npz'):
            data = np.load(file_path, allow_pickle=True)
            return data['data'], data['labels']
    
    # Tải dữ liệu
    X_train, y_train = [], []
    for file in tqdm(train_files, desc="Đang tải dữ liệu train"):
        X, y = load_file(file)
        X_train.append(X)
        y_train.append(y)
    
    X_val, y_val = [], []
    for file in tqdm(val_files, desc="Đang tải dữ liệu validation"):
        X, y = load_file(file)
        X_val.append(X)
        y_val.append(y)
    
    X_test, y_test = [], []
    for file in tqdm(test_files, desc="Đang tải dữ liệu test"):
        X, y = load_file(file)
        X_test.append(X)
        y_test.append(y)
    
    # Gộp dữ liệu
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    X_val = np.vstack(X_val)
    y_val = np.concatenate(y_val)
    
    X_test = np.vstack(X_test)
    y_test = np.concatenate(y_test)
    
    print(f"Kích thước X_train: {X_train.shape}")
    print(f"Kích thước y_train: {y_train.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def prepare_data_for_cnn(X, y, input_shape=(6, 7680, 1)):
    """
    Chuẩn bị dữ liệu cho mô hình CNN_Resnet50 sử dụng PyTorch
    
    Parameters:
    -----------
    X: numpy.ndarray
        Dữ liệu đầu vào, shape (n_samples, n_channels, n_times)
    y: numpy.ndarray
        Nhãn, shape (n_samples,)
    input_shape: tuple
        Kích thước đầu vào cho CNN
        
    Returns:
    --------
    X_tensor: torch.Tensor
        Dữ liệu đã được định dạng lại cho CNN
    y_tensor: torch.Tensor
        Nhãn được chuyển đổi thành tensor
    """
    # Reshape dữ liệu cho phù hợp với đầu vào của CNN
    n_samples, n_channels, n_times = X.shape
    target_channels, target_times, _ = input_shape
    
    # Nếu số kênh khác với target_channels, cần điều chỉnh
    if n_channels != target_channels:
        print(f"Cảnh báo: Số kênh ({n_channels}) khác với số kênh trong input_shape ({target_channels})")
        
        if n_channels < target_channels:
            # Padding thêm các kênh nếu thiếu
            padding = np.zeros((n_samples, target_channels - n_channels, n_times))
            X = np.concatenate([X, padding], axis=1)
        else:
            # Chỉ lấy số kênh cần thiết nếu dư
            X = X[:, :target_channels, :]
    
    # Nếu số điểm thời gian khác với target_times, cần điều chỉnh
    if n_times != target_times:
        print(f"Cảnh báo: Số điểm thời gian ({n_times}) khác với số điểm thời gian trong input_shape ({target_times})")
        
        if n_times < target_times:
            # Padding thêm các điểm thời gian nếu thiếu
            padding = np.zeros((n_samples, target_channels, target_times - n_times))
            X = np.concatenate([X, padding], axis=2)
        else:
            # Chỉ lấy số điểm thời gian cần thiết nếu dư
            X = X[:, :, :target_times]
    
    # Chuyển đổi numpy array sang PyTorch tensor
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    
    return X_tensor, y_tensor

if __name__ == "__main__":
    # Ví dụ tải dữ liệu đã tiền xử lý
    processed_dir = "D:/codeN/Biomedical/Dataset/ds005555_processed"
    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_dataset(
        processed_dir=processed_dir,
        subjects=5,  # Chỉ tải 5 subjects đầu
        test_size=0.2,
        val_size=0.2
    )
    
    X_train_torch, y_train_torch = prepare_data_for_cnn(X_train, y_train)
    X_val_torch, y_val_torch = prepare_data_for_cnn(X_val, y_val)
    X_test_torch, y_test_torch = prepare_data_for_cnn(X_test, y_test)
    
    print(f"Kích thước X_train_torch: {X_train_torch.shape}")
    print(f"Kích thước y_train_torch: {y_train_torch.shape}")
