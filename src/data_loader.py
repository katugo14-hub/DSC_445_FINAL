import pickle
from pathlib import Path


def load_subject_pkl(file_path):
    """
    Load one PPG-DaLiA subject .pkl file.
    """
    file_path = Path(file_path)

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    ppg = data["signal"]["wrist"]["BVP"]
    acc = data["signal"]["wrist"]["ACC"]
    hr = data["label"]

    return ppg, acc, hr