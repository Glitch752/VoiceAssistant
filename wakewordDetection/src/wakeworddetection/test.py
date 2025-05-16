import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
from collections import deque
import time
import threading

SAMPLE_RATE = 16000
WINDOW_DURATION = 1.0  # seconds
STRIDE_DURATION = 0.25  # seconds

WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)
STRIDE_SIZE = int(SAMPLE_RATE * STRIDE_DURATION)

MODEL_PATHS = [
    ("Normal", './processed/wakeword_model.tflite'),
    # ("Optimized", './processed/wakeword_model_v2_optimized.tflite')
]

DEBOUNCE_TIME = 0.5  # seconds

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=512, hop_length=256)
    mfcc = mfcc.T  # (frames, 13)
    if mfcc.shape[0] < 32:
        mfcc = np.pad(mfcc, ((0, 32 - mfcc.shape[0]), (0, 0)))
    return mfcc[:32, :]  # (32, 13)


def test_model_with_path(path, name):
    interpreter = tf.lite.Interpreter(path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Audio buffer
    buffer = deque(maxlen=WINDOW_SIZE)
    
    last_detection_time = 0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio error:", status)

        buffer.extend(indata[:, 0])  # mono

        if len(buffer) == WINDOW_SIZE:
            detect_wakeword()
    
    def detect_wakeword():
        audio_np = np.array(buffer, dtype=np.float32)
        features = extract_features(audio_np)
        input_tensor = features[np.newaxis, ..., np.newaxis].astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        if prediction > 0.7:
            nonlocal last_detection_time
            current_time = time.time()
            if current_time - last_detection_time > DEBOUNCE_TIME:
                last_detection_time = current_time
                print(f"Wakeword detected by {name}! Confidence: {prediction:.3f}")
    
    print("Listening for wakeword...")
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=STRIDE_SIZE, callback=audio_callback):
        while True:
            time.sleep(0.1)

def test_model():
    threads: list[threading.Thread] = []
        
    for name, path in MODEL_PATHS:
        print(f"Testing model: {name}")
        # Spawn a thread so we can run multiple models at once
        thread = threading.Thread(target=test_model_with_path, args=(path, name))
        thread.daemon = True
        
        thread.start()
        threads.append(thread)
        
    while True:
        time.sleep(1)