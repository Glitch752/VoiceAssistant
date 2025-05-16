import os
import shutil
import librosa
import numpy as np
import soundfile as sf
import random

NOISE_DIR = './data/background/'

WAKEWORD_DIR = './data/wakeword/'
NOT_WAKEWORD_DIR = './data/not_wakeword/'

AUGMENTED_WAKEWORD_DIR = './data/augmented_wakeword/'
AUGMENTED_NOT_WAKEWORD_DIR = './data/augmented_not_wakeword/'

def load_noise_files(max=500):
    noise_files = []
    files = os.listdir(NOISE_DIR)
    files = random.sample(files, min(max, len(files)))
    
    for i, fname in enumerate(files):
        print(f"Loading noise file {i+1}/{len(files)}: {fname}")
        
        if fname.endswith('.wav'):
            y, _ = librosa.load(os.path.join(NOISE_DIR, fname), sr=16000)
            # A lot of the noise files are extremely loud, so we normalize them to a relatively low level
            y = y / np.max(np.abs(y))
            # Normalize to -10 dBFS
            y = y * 10**(-10 / 20)
            
            noise_files.append(y)
    return noise_files

def add_background_noise(y, noises, signal_to_noise_ratio_db=20):
    noise = random.choice(noises)
    if len(noise) < len(y):
        noise = np.pad(noise, (0, len(y) - len(noise)))
    else:
        # Pick a random subsection of the noise
        start = random.randint(0, len(noise) - len(y))
        noise = noise[start:start + len(y)]
    
    signal_power = np.mean(y ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(signal_power / (10**(signal_to_noise_ratio_db / 10) * noise_power))
    noisy = y + scale * noise
    return noisy

def time_stretch(y):
    rate = random.uniform(0.9, 1.0)
    return librosa.effects.time_stretch(y, rate=rate)

# Seems to reduce quality
# def pitch_shift(y, sr):
#     steps = random.random() * 2 - 1  # Random shift between -1 and 1 semitones
#     return librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

def change_volume(y):
    # Randomly change volume between -10 dB and +5 dB
    volume_change = random.uniform(-10, 5)
    y = y * 10**(volume_change / 20)
    return y

def augment_sample(y, sr, noises):
    y_aug = np.copy(y)
    
    if random.random() < 0.4:
        y_aug = time_stretch(y_aug)
    
    # if random.random() < 0.4:
    #     y_aug = pitch_shift(y_aug, sr)
    
    if random.random() < 0.4:
        y_aug = change_volume(y_aug)
    
    if random.random() < 0.8:
        y_aug = add_background_noise(y_aug, noises)
    
    y_aug = librosa.util.fix_length(y_aug, size=16000)
    return y_aug

def augment(noises, input_dir, output_dir, n_augment):
    os.makedirs(output_dir, exist_ok=True)
    
    # Delete all files in the augmented directory
    for fname in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, fname))

    print(f"Augmenting wakeword files in {input_dir} into {output_dir}")
    
    for fname in os.listdir(input_dir):
        if not fname.endswith('.wav'):
            continue
        
        y, sr = librosa.load(os.path.join(input_dir, fname), sr=16000)
        
        # Normalize the original file to -10 dBFS
        y = y / np.max(np.abs(y))
        y = y * 10**(-10 / 20)
        
        for i in range(n_augment):
            y_aug = augment_sample(y, sr, noises)
            out_name = f"{fname[:-4]}_aug{i}.wav"
            sf.write(os.path.join(output_dir, out_name), y_aug, sr)
        
        # Also copy the original file
        out_name = f"{fname[:-4]}_orig.wav"
        shutil.copy(os.path.join(input_dir, fname), os.path.join(output_dir, out_name))
        print(f"Augmented {fname} into {n_augment} times and copied original.")
    
    print(f"Done augmenting into {output_dir}")


def augment_wakewords(n_augment=10):
    print("Loading noise files...")
    noises = load_noise_files()
    
    print("Augmenting wakeword samples...")
    augment(noises, WAKEWORD_DIR, AUGMENTED_WAKEWORD_DIR, n_augment)
    print("Augmenting not wakeword samples...")
    augment(noises, NOT_WAKEWORD_DIR, AUGMENTED_NOT_WAKEWORD_DIR, n_augment)
    