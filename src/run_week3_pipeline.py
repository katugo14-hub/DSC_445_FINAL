import time
from pathlib import Path
import torch

#Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT/"data"
PROCESSED_DIR = DATA_DIR/"processed_data"
PROCESSED_DIR.mkdir(exist_ok=True)


#Deep learning modules
from dl_preprocessing import preprocess_all_subjects
from train_dl import tune_hyperparameters, train_loso
from models_dl import CNNModel
from sanity_checks import(
    check_data_shapes,
    check_for_nans,
    check_model_forward_pass
)

def _processed_exists():
    """check if processed_data/already contains subject windows."""
    subjects = sorted({p.name.split("_")[0] for p in PROCESSED_DIR.glob("*_X.npy")})
    return subjects if subjects else None

                       
def main(skip_preprocess:bool=False, skip_tuning:bool=False):
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #Step1: Preprocessing
    print("\n---Step 1/4: Preprocessing-(PPG + ACC + HR)--")

    if skip_preprocess:
        subjects = _processed_exists()
        if subjects is None:
            raise RuntimeError("No processed data found. Cannot skip preprocessing.")
        print("[skip] Using cached processed_data/")
    else:
        preprocess_all_subjects()
        subjects = _processed_exists()
        print("[done] Preprocessing complete.")

    #Step2: Sanity Checks
    print("\n---Step 2/4: Sanity Checks---")
    check_data_shapes()
    check_for_nans()
    check_model_forward_pass()

    #Step3: Hyperparameter Tuning
    print("\n---Step 3/4: Hyperparameter Tuning (Validation Subject: S2)---")

    if skip_tuning:
        raise NotImplementedError("Skipping tunning is not supported yet.")
    
    best_hparams = tune_hyperparameters(
        model_class=CNNModel,
        val_subject="S2",
        device=device
    )
    print("[tuning] Best hyperparameters:", best_hparams)

    #Step4: LOSO Training
    print("\n---Step 4/4: LOSO Training---")

    loso_results = train_loso(
        model_class=CNNModel,
        hparams=best_hparams,
        device=device
    )

    #Summary
    print("\n---Summary---")
    print("Final LOSO Results:")
    print(loso_results)

    elapsed = time.time() - t0
    print(f"\n[done] Week 3 pipeline finished in {elapsed:.1f}s")
    
    

if __name__ == "__main__":
    main()
