import time
import sounddevice as sd
import soundfile as sf
import os
import threading
import numpy as np

WAKEWORD_SAVE_PATH = './data/wakeword/'
NOT_WAKEWORD_SAVE_PATH = './data/not_wakeword/'
    
def record():
    os.makedirs(WAKEWORD_SAVE_PATH, exist_ok=True)

    number_of_recordings = 50
    
    # Find the last recording number
    existing_files = os.listdir(WAKEWORD_SAVE_PATH)
    existing_files = [f for f in existing_files if f.endswith('.wav')]
    existing_numbers = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
    start = max(existing_numbers) + 1 if existing_numbers else 0
    print(f"Starting from recording number {start}.")

    for i in range(number_of_recordings):
        input(f"Press Enter to start recording #{i+1}/{number_of_recordings}...")
        # Wait to reduce keyboard noise
        time.sleep(0.3)
        
        print(f"Say the wakeword now!")
        recording = sd.rec(int(16000), samplerate=16000, channels=1)
        sd.wait()
        sf.write(f"{WAKEWORD_SAVE_PATH}/wake_{i + start}.wav", recording, 16000)

def record_negatives():
    os.makedirs(NOT_WAKEWORD_SAVE_PATH, exist_ok=True)

    # Find the last recording number
    existing_files = os.listdir(NOT_WAKEWORD_SAVE_PATH)
    existing_files = [f for f in existing_files if f.endswith('.wav')]
    existing_numbers = [int(f.split('_')[-1].split('.')[0]) for f in existing_files]
    start_idx = max(existing_numbers) + 1 if existing_numbers else 0
    print(f"Starting from recording number {start_idx}.")

    print("Press Enter to stop recording negative samples.")
    recording = []

    def input_thread():
        input()  # wait for Enter to stop recording
        nonlocal stop_recording
        stop_recording = True

    stop_recording = False
    thread = threading.Thread(target=input_thread)
    thread.start()

    with sd.InputStream(samplerate=16000, channels=1, dtype='float32') as stream:
        while not stop_recording:
            chunk, _ = stream.read(16000)  # 1 second chunks
            recording.append(chunk)

    audio = np.concatenate(recording).squeeze()
    print(f"Recorded {len(audio)/16000:.2f} seconds of audio.")

    # Now segment it into overlapping 1s windows with 0.5s stride
    stride = 0.5
    window = 1.0

    index = 0
    start = 0
    end = int(window * 16000)
    
    while end <= len(audio):
        segment = audio[start:end]
        sf.write(f"{NOT_WAKEWORD_SAVE_PATH}/not_wakeword_{index + start_idx}.wav", segment, 16000)
        index += 1
        start += int(stride * 16000)
        end = start + int(window * 16000)

    print(f"Saved {index} negative segments.")
    
def get_char():
    import msvcrt
    return msvcrt.getch().decode('utf-8', errors='ignore')

def validate():
    # Remove .bak from the extension of files with it
    for filename in os.listdir(WAKEWORD_SAVE_PATH):
        if filename.endswith(".wav.bak"):
            os.rename(os.path.join(WAKEWORD_SAVE_PATH, filename), os.path.join(WAKEWORD_SAVE_PATH, filename[:-4]))
    
    files = os.listdir(WAKEWORD_SAVE_PATH)
    # Sort based on numbers in the filename
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Play back each recording and allow the user to pick keep/delete with space or del
    for filename in files:
        if filename.endswith(".wav"):
            filepath = os.path.join(WAKEWORD_SAVE_PATH, filename)
            
            # For some reason, soundfile crops clips oddly? Just load manually.
            import wave
            with wave.open(filepath, 'rb') as wf:
                data = wf.readframes(wf.getnframes())
                # Convert to numpy array
                import numpy as np
                data = np.frombuffer(data, dtype=np.int16)
                data = data.reshape(-1, 1)
                # Play the audio. We use a loop to avoid closing the stream which acts funky
                sd.play(data, samplerate=16000, loop=True)
            
            print(f"Played {filename}. Press space to keep or delete to remove. Press C to exit.")
            if get_char() == ' ':
                print(f"Kept {filename}.")
            elif get_char() == 'c':
                print("Exiting validation.")
                break
            else:
                os.rename(filepath, filepath + '.bak')
                print(f"Removed {filename}.")