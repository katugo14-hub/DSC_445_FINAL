import numpy as np
from dl_dataset import get_all_subjects, load_subject
from models_dl import CNNModel
import torch


def check_data_shapes():
    subjects = get_all_subjects()
    print("Subjects found:", subjects)

    X, y = load_subject(subjects[0])
    print("X shape:", X.shape)     #(num_windows, 512, 4)
    print("y shape:", y.shape)     #(num_windows,)


def check_for_nans():
    subjects = get_all_subjects()
    X, y = load_subject(subjects[0])

    print("NaNs in X:", np.isnan(X).sum())
    print("NaNs in y:", np.isna(y).sum())


def check_model_forward_pass():
    model = CNNModel()
    dummy = torch.randn(2, 512, 4)    #batch=2
    out = model(dummy)
    print("Model output shape:", out.shape)


if __name__ == "__main__":
    check_data_shapes()
    check_for_nans()
    check_model_forward_pass()
