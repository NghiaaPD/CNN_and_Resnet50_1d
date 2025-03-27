import os
import numpy as np
import pandas as pd
import glob
import h5py
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from Utils import read_edf_file, filter_band, remove_artifacts_ica

def process_subject(subject_dir, output_dir, use_hdf5=True):
    """
    Xử lý dữ liệu EEG cho một subject
    
    Parameters:
    -----------
    subject_dir: str
        Đường dẫn đến thư mục subject
    output_dir: str
        Đường dẫn đến thư mục đầu ra
    use_hdf5: bool
        Sử dụng định dạng HDF5 nếu True, ngược lại sử dụng NPZ
    """
    try:
        subject_id = os.path.basename(subject_dir)
        print(f"Đang xử lý {subject_id}...")

        eeg_dir = os.path.join(subject_dir, "eeg")
        if not os.path.exists(eeg_dir):
            print(f"Thư mục EEG không tồn tại cho {subject_id}")
            eeg_dir = subject_dir

        edf_files = glob.glob(os.path.join(eeg_dir, "*.edf"))

        if edf_files:
            print(f"Các file EDF tìm thấy: {edf_files}")
        else:
            print(f"Không tìm thấy file EDF nào trong {eeg_dir}")

            all_files = glob.glob(os.path.join(subject_dir, "**/*.*"), recursive=True)
            print(f"Một số file trong thư mục subject: {all_files[:5] if all_files else 'Không có file nào'}")
            return

        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        for edf_file in edf_files:
            try:
                file_basename = os.path.basename(edf_file).replace('_eeg.edf', '')
                events_file = edf_file.replace('_eeg.edf', '_events.tsv')
                
                if not os.path.exists(events_file):
                    print(f"File events không tồn tại: {events_file}")
                    continue

                print(f"Đang đọc {edf_file}...")
                raw = read_edf_file(edf_file)

                print(f"Đang lọc tín hiệu...")
                raw = filter_band(raw, l_freq=1, h_freq=50, notch_freq=50)

                try:
                    print(f"Đang áp dụng ICA...")
                    n_channels = len(raw.ch_names)
                    n_components = min(n_channels - 1, 5)
                    raw_cleaned = remove_artifacts_ica(raw, n_components=n_components)
                except Exception as e:
                    print(f"Lỗi khi áp dụng ICA: {e}. Sử dụng dữ liệu đã lọc.")
                    raw_cleaned = raw

                data = raw_cleaned.get_data()
                sfreq = raw_cleaned.info['sfreq']
                ch_names = raw_cleaned.ch_names

                events_data = pd.read_csv(events_file, sep='\t')
                labels = events_data['stage_hum'].values

                epoch_samples = int(30 * sfreq)

                epochs = []
                epoch_labels = []
                
                for i in range(len(labels)):
                    start_sample = i * epoch_samples
                    end_sample = start_sample + epoch_samples
                    
                    if end_sample <= data.shape[1]:
                        epoch = data[:, start_sample:end_sample]

                        if labels[i] != 8:
                            epochs.append(epoch)
                            epoch_labels.append(labels[i])
                
                X = np.array(epochs)
                y = np.array(epoch_labels)
                
                print(f"Đã xử lý {len(epochs)} epoch cho {file_basename}")
                
                if use_hdf5:
                    output_file = os.path.join(subject_output_dir, f"{file_basename}_processed.h5")
                    with h5py.File(output_file, 'w') as f:
                        f.create_dataset('data', data=X.astype(np.float32), compression="gzip", compression_opts=9)
                        f.create_dataset('labels', data=y)
                        f.create_dataset('sfreq', data=sfreq)
                        
                        dt = h5py.special_dtype(vlen=str)
                        ch_names_ds = f.create_dataset('ch_names', (len(ch_names),), dtype=dt)
                        for i, name in enumerate(ch_names):
                            ch_names_ds[i] = name
                else:
                    output_file = os.path.join(subject_output_dir, f"{file_basename}_processed.npz")
                    np.savez_compressed(
                        output_file,
                        data=X.astype(np.float32),  # Dùng float32 tiết kiệm dung lượng
                        labels=y,
                        sfreq=sfreq,
                        ch_names=np.array(ch_names, dtype=object)
                    )
                
                print(f"Đã lưu dữ liệu đã xử lý vào {output_file}")
                
            except Exception as e:
                print(f"Lỗi khi xử lý file {edf_file}: {e}")
                
    except Exception as e:
        print(f"Lỗi khi xử lý subject {subject_dir}: {e}")

def preprocess_dataset(base_dir, output_dir, n_jobs=None, use_hdf5=True):
    """
    Tiền xử lý toàn bộ dataset
    
    Parameters:
    -----------
    base_dir: str
        Đường dẫn đến thư mục dữ liệu gốc
    output_dir: str
        Đường dẫn đến thư mục đầu ra
    n_jobs: int, optional
        Số lượng tiến trình song song, mặc định là None (sử dụng tất cả CPU)
    use_hdf5: bool
        Sử dụng định dạng HDF5 nếu True, ngược lại sử dụng NPZ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    subject_dirs = sorted(glob.glob(os.path.join(base_dir, "sub-*")))
    
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    print(f"Tiền xử lý dataset với {n_jobs} tiến trình song song...")
    print(f"Định dạng lưu trữ: {'HDF5' if use_hdf5 else 'NPZ'}")
    
    # Xử lý các subject song song
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for subject_dir in subject_dirs:
            executor.submit(process_subject, subject_dir, output_dir, use_hdf5)

if __name__ == "__main__":
    base_dir = "D:/codeN/Biomedical/Dataset/ds005555"
    processed_dir = "D:/codeN/Biomedical/Paper/RL-CNN_Resnet50/data"

    preprocess_dataset(
        base_dir=base_dir,
        output_dir=processed_dir,
        n_jobs=4,  # Sử dụng 4 CPU
        use_hdf5=True  # Sử dụng định dạng HDF5
    )
    
