import numpy as np
import soundfile as sf
import os

OUT_DIR = './data/silence/'

def generate_silence(samples=1000):
    os.makedirs(OUT_DIR, exist_ok=True)

    for i in range(samples):
        amplitude = 10 ** np.random.uniform(-6, -3)
        
        silent_audio = np.random.normal(0, amplitude, 16000).astype(np.float32)
        filename = os.path.join(OUT_DIR, f"silence_{i}.wav")
        sf.write(filename, silent_audio, samplerate=16000)

    print(f"Generated {samples} silent samples in {OUT_DIR}")