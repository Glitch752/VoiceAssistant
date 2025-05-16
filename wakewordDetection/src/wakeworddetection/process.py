import os
import librosa
import numpy as np
import soundfile as sf

DATA_DIR = './data/'
OUT_DIR = './processed/'
os.makedirs(OUT_DIR, exist_ok=True)

def process_audio(file_path, max_len=16000):
    y, sr = librosa.load(file_path, sr=16000)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.T  # shape: (time_steps, 13)

def process_dataset():
    print("Processing dataset...")
    X, y = [], []
    for label, subfolder in [(0, 'silence'), (0, 'not_wakeword_from_dataset'), (0, 'augmented_not_wakeword'), (1, 'augmented_wakeword')]:
        folder_path = os.path.join(DATA_DIR, subfolder)
        print(f"Processing folder: {folder_path}")
        
        for fname in os.listdir(folder_path):
            if fname.endswith('.wav'):
                path = os.path.join(folder_path, fname)
                mfcc = process_audio(path)
                X.append(mfcc)
                y.append(label)
                
                print(f"Processed {fname}, shape: {mfcc.shape}")
    
    X = np.array(X)
    y = np.array(y)
    
    np.savez(os.path.join(OUT_DIR, 'data.npz'), X=X, y=y)
    print(f"Saved {len(X)} samples to {OUT_DIR}data.npz")