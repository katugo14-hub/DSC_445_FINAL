import numpy as np
import os

def load_subject(subject_name, data_dir="processed_data"):
    """ Loads on e subject's processed DL data.
    Returns:
          X: (num_windows, 512, 4)
          y: (num_windoes,)
    """
    X = np.load(os.path.join(data_dir, f"{subject_name}_X.npy"))
    y = np.load(os.path.join(data_dir, f"{subject_name}_y.npy"))
    return X, y

def get_all_subjects(data_dir="processed_data"):
    """ Returns a sorted list of all subjetcs avaialle in processed_data"""
    files = os.listdir(data_dir)
    subjects = sorted(list({f.split("_")[0] for f in files if f.endswith("_X.npy")}))
    return subjects

def get_loso_split(test_subject, all_subjects, data_dir="processed_data"):
    """
    LOSO split loader.
    Loads test subjetc normally.
    Loads all training subjects using memory mapping to avoid RAM overload.

    Returns: X_train, y_train, X_test, y_test
    """

    #load test subject fully
    X_test = np.load(os.path.join(data_dir, f"{test_subject}_X.npy"))
    y_test = np.load(os.path.join(data_dir, f"{test_subject}_y.npy"))

    #memory-efficient loading for training subjetcs
    X_train_list = []
    y_train_list = []

    for subject in all_subjects:
        if subject == test_subject:
            continue

        X_train_list.append(np.load(os.path.join(data_dir, f"{subject}_X.npy"),
                                    mmap_mode="r"))
        y_train_list.append(np.load(os.path.join(data_dir, f"{subject}_y.npy"),
                                    mmap_mode="r"))

    #concatenate into full training arrays
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    return X_train, y_train, X_test, y_test

#optional: PyTorch dataset wrapper
try:
    import torch
    from torch.utils.data import Dataset
    
    class WindowDataset(Dataset):
        """converts numpy windows into a PyTorch-friendly dataset"""

        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
except ImportError:
    #if PyTorch isn't installed yet, skip the Dataset class
    pass
    
    

                             
