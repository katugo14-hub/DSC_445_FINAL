import warnings
import pickle
import os
import numpy as np
import glob
import gc

from scipy.signal import resample, butter, filtfilt




#suppress NumPy warning (PPG-DaLiA dataset created & "pickled" several years ago using an older version of NumPy)
warnings.filterwarnings("ignore", message=r".*dtype\(\): align should be passed.*")

#Set path of dataset folder
data_dir = r"C:\Users\Smith\OneDrive\Desktop\ML Class Project\PPG_DaLiA_Project\data"
output_dir = "processed_data"
os.makedirs(output_dir, exist_ok=True)

subject_files = sorted(glob.glob(os.path.join(data_dir, "S*.pkl")))


#Bandpass filter PPG 
fs = 64     #sampling rate

#design butterworth bandpass filter
lowcut = 0.5
highcut = 4.0
order = 4

b, a = butter(order, [lowcut/(fs/2), highcut/(fs/2)], btype='band')

# Windowing parameters
window_size = 512
step_size = 128


for file_path in subject_files:
    subject_name = os.path.basename(file_path).replace(".pkl", "")
    print(f"---Processing {subject_name} ---")

    #load the pickle file using latin1 encoding
    with open (file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")


    #extract signals (PPG and ACC)
    curr_ppg = data["signal"]["wrist"]["BVP"]       # PPG signal
    curr_acc = data["signal"]["wrist"]["ACC"]      # Accelerometer

    # extract HR label
    curr_label = data["label"]

    #Preprocessing
    #Resample ACC from 32Hz to 64 Hz to match PPG length
    ppg_len = curr_ppg.shape[0]        # target length

    #resample ACC to match PPG length
    acc_resampled = resample(curr_acc, ppg_len)

    #filter PPG
    ppg_filtered = filtfilt(b, a, curr_ppg.squeeze())      #remove the extra dimension

    #Normalize PPG and ACC (Z-score)
    #Normalize PPG (1D)
    ppg_norm = (ppg_filtered - np.mean(ppg_filtered)) / np.std(ppg_filtered)

    #Normalize ACC (each axis separately)
    acc_norm = (acc_resampled - np.mean(acc_resampled, axis=0)) / np.std(acc_resampled, axis=0)

    #combine PPG, ACC channels into a single matrix (with 4 channel signal)
    X = np.column_stack([ppg_norm, acc_norm]).astype(np.float32)


    #Windowing
    sub_windows = []
    sub_hr = []

    for start in range(0, len(X) - window_size, step_size):
        end = start + window_size
        sub_windows.append(X[start:end])


        #HR alignment: labels are 2Hz ( 1 label every 32 PPG samples)
        hr_idx = end // 32
        if hr_idx < len(curr_label):
            sub_hr.append(curr_label[hr_idx])
        else:
            sub_hr.append(curr_label[-1])           # fallback for last window


    #save to disk and clear RAM
    np.save(os.path.join(output_dir, f"{subject_name}_X.npy"), np.array(sub_windows, dtype=np.float32))
    np.save(os.path.join(output_dir, f"{subject_name}_y.npy"), np.array(sub_hr, dtype=np.float32))

    print(f"Successfully saved {len(sub_windows)} windows.")
    print("-" * 30)

    #free memory after storing (to avoid memory error)
    del data, curr_ppg, curr_acc, curr_label, X, ppg_norm, acc_norm
    gc.collect()

print("\nProcessing complete.")





# LOSO (leave-one-subject-out) 
def get_loso_split (test_subject_name, all_subject_names):
    """
    Loads one fold of LOSO by reading from the processed_data folder.
    uses memory mapping to avoid RAM overload.
    """

    #Load Test subject
    X_test = np.load(f"processed_data/{test_subject_name}_X.npy")
    y_test = np.load(f"processed_data/{test_subject_name}_y.npy")

    #load train subjects using memory mapping
    X_train_list = []
    y_train_list = []

    # prepare Training data pointers
    for s in all_subject_names:
        if s != test_subject_name:
            #mmap_moder='r' allows accessing the data without loading it all at once
            X_train_list.append(np.load(f"processed_data/{s}_X.npy", mmap_mode='r'))
            y_train_list.append(np.load(f"processed_data/{s}_y.npy", mmap_mode='r'))

         
    #concatenate
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    return X_train, y_train, X_test, y_test



# Model training
all_subs = [f"S{i}" for i in range(1, 16)] # List of all subject IDs

for test_sub in all_subs:
    if not os.path.exists(f"{output_dir}/{test_sub}_X.npy"):
        continue

    print(f"Training LOSO Fold: Testing on {test_sub}")
    X_train, y_train, X_test, y_test = get_loso_split(test_sub, all_subs)

    print(f"Train Shape: {X_train.shape} | Test Shape: {X_test.shape}")
    
    
    
        


        
        
     




              

