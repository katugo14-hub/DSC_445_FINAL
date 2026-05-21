import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from dl_dataset import get_loso_split, get_all_subjects, WindowDataset
from models_dl import CNNModel, CNNLSTMModel


# Train for one epoch
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X).squeeze()
        loss = criterion(preds, y)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


#validate
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            total_loss += loss.item()

    return total_loss / len(loader)


#Hyperparameter tuning using ONE validation subject
def tune_hyperparameters(model_class, val_subject="S2", device="cpu"):

    all_subjects = get_all_subjects()
    X_train, y_train, X_val, y_val = get_loso_split(val_subject, all_subjects)

    #convert to PyTorch datasets
    train_ds = WindowDataset(X_train, y_train)
    val_ds = WindowDataset(X_val, y_val)

    #Hyperparameter search space
    learning_rates = [1e-4, 3e-4, 1e-3]
    dropouts = [0.0, 0.2, 0.5]
    batch_sizes = [8, 16, 32]
    filter_sizes = [32, 64]
    kernel_sizes = [3,5,7]

    best_val_loss = float("inf")
    best_hparams = None

    criterion = nn.L1Loss()     #MAE

    for lr in learning_rates:
        for dropout in dropouts:
            for bs in batch_sizes:
                for nf in filter_sizes:
                    for ks in kernel_sizes:

                        print(f"Tuning: lr={lr}, dropout={dropout}, batch={bs}, filters={nf}, ks={ks}")

                        model = model_class(num_filters=nf, dropout=dropout, kernel_size=ks).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
                        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

                        #Train for a few epochs
                        for epoch in range(3):
                            train_one_epoch(model, train_loader, optimizer, criterion, device)

                        #validate
                        val_loss = validate(model, val_loader, criterion, device)

                        print(f"Validation MAE: {val_loss:.4f}")

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_hparams = {
                                "lr": lr,
                                "dropout": dropout,
                                "batch_size": bs,
                                "num_filters": nf,
                                "kernel_size": ks
                            }
    print("\nBest Hyperparameters Found:")
    print(best_hparams)


    return best_hparams


#Full LOSO training using fixed hyperparameters
def train_loso(model_class, hparams, device="cpu"):

    all_subjects = get_all_subjects()
    criterion = nn.L1Loss()    #MAE

    results = {}

    for test_sub in all_subjects:
        print(f"\n--- LOSO Fold: Testing on {test_sub} ---")

        X_train, y_train, X_test, y_test = get_loso_split(test_sub, all_subjects)

        train_ds = WindowDataset(X_train, y_train)
        test_ds = WindowDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=hparams["batch_size"], shuffle=False)

        #Build model with tuned hyperparameters
        model = model_class(
            num_filters = hparams["num_filters"],
            dropout = hparams["dropout"],
            kernel_size = hparams["kernel_size"]
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])

        #Train for fixed epochs
        for epoch in range(5):
            train_one_epoch(model, train_loader, optimizer, criterion, device)

        #Evaluate
        test_loss = validate(model, test_loader, criterion, device)
        results[test_sub] = test_loss

        print(f"Test MAE for {test_sub}: {test_loss:.4f}")

    print("\n---Final LOSO Results---")
    print(results)

    return results











        






















                                




























    






















            
