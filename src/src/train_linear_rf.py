import glob
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import extract_features_batch
from linear_rf_windows import create_subject_windows


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 123


def load_all_subject_features():
    """
    Loads all subjects, creates windows, extracts features,
    and stores them in a dictionary.
    """

    subject_files = sorted(glob.glob(str(DATA_DIR / "S*.pkl")))

    print("Subjects found:", len(subject_files))

    if len(subject_files) == 0:
        raise FileNotFoundError(
            "No S*.pkl files found. Put the dataset inside the data/ folder."
        )

    subject_data = {}

    for file_path in subject_files:
        subject_name = Path(file_path).stem

        print("\n" + "=" * 60)
        print("Processing:", subject_name)
        print("=" * 60)

        X_ppg, X_acc, y = create_subject_windows(file_path)

        print("PPG windows:", X_ppg.shape)
        print("ACC windows:", X_acc.shape)
        print("HR labels  :", y.shape)

        X_feat = extract_features_batch(
            X_ppg,
            X_acc,
            n_workers=1,
            verbose_every=2000
        )

        print("Feature matrix:", X_feat.shape)

        subject_data[subject_name] = {
            "X": X_feat,
            "y": y
        }

        gc.collect()

    return subject_data


def evaluate_regression(y_true, y_pred):
    """
    Computes regression evaluation metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def run_loso_linear_rf(subject_data):
    """
    Runs Leave-One-Subject-Out evaluation using
    Linear Regression and Random Forest Regression.
    """

    subject_names = list(subject_data.keys())

    results = []

    for test_subject in subject_names:
        print("\n" + "=" * 70)
        print("Testing on:", test_subject)
        print("=" * 70)

        X_test = subject_data[test_subject]["X"]
        y_test = subject_data[test_subject]["y"]

        X_train_list = []
        y_train_list = []

        for subject in subject_names:
            if subject != test_subject:
                X_train_list.append(subject_data[subject]["X"])
                y_train_list.append(subject_data[subject]["y"])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        print("X_train:", X_train.shape)
        print("X_test :", X_test.shape)

        linear_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])

        linear_model.fit(X_train, y_train)
        linear_pred = linear_model.predict(X_test)

        linear_mae, linear_rmse, linear_r2 = evaluate_regression(
            y_test,
            linear_pred
        )

        print("\nLinear Regression")
        print("MAE :", round(linear_mae, 3))
        print("RMSE:", round(linear_rmse, 3))
        print("R2  :", round(linear_r2, 3))

        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        rf_mae, rf_rmse, rf_r2 = evaluate_regression(y_test, rf_pred)

        print("\nRandom Forest Regression")
        print("MAE :", round(rf_mae, 3))
        print("RMSE:", round(rf_rmse, 3))
        print("R2  :", round(rf_r2, 3))

        results.append({
            "Test Subject": test_subject,
            "Linear Regression MAE": linear_mae,
            "Linear Regression RMSE": linear_rmse,
            "Linear Regression R2": linear_r2,
            "Random Forest MAE": rf_mae,
            "Random Forest RMSE": rf_rmse,
            "Random Forest R2": rf_r2,
        })

    return pd.DataFrame(results)


def save_outputs(results_df):
    """
    Saves CSV results and comparison plots.
    """

    results_path = OUTPUT_DIR / "linear_rf_loso_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nSaved results to:", results_path)

    models = ["Linear Regression", "Random Forest"]

    mae_values = [
        results_df["Linear Regression MAE"].mean(),
        results_df["Random Forest MAE"].mean(),
    ]

    rmse_values = [
        results_df["Linear Regression RMSE"].mean(),
        results_df["Random Forest RMSE"].mean(),
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(models, mae_values)
    plt.ylabel("Average MAE")
    plt.title("Average MAE Comparison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_rf_mae_comparison.png", dpi=300)
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.bar(models, rmse_values)
    plt.ylabel("Average RMSE")
    plt.title("Average RMSE Comparison")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "linear_rf_rmse_comparison.png", dpi=300)
    plt.show()


def main():
    subject_data = load_all_subject_features()

    results_df = run_loso_linear_rf(subject_data)

    print("\nFINAL LOSO RESULTS")
    print(results_df)

    print("\nAVERAGE RESULTS")
    print(results_df.mean(numeric_only=True))

    save_outputs(results_df)


if __name__ == "__main__":
    main()
