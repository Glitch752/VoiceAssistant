import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m wakeworddetection <command>")
        return
    
    if sys.argv[1] == "record":
        from wakeworddetection.record import record
        record()
    elif sys.argv[1] == "record_negatives":
        from wakeworddetection.record import record_negatives
        record_negatives()
    elif sys.argv[1] == "validate":
        from wakeworddetection.record import validate
        validate()
    elif sys.argv[1] == "process":
        from wakeworddetection.process import process_dataset
        process_dataset()
    elif sys.argv[1] == "augment":
        from wakeworddetection.augment import augment_wakewords
        augment_wakewords()
    elif sys.argv[1] == "load_datasets":
        from wakeworddetection.load_datasets import load_datasets
        load_datasets()
    elif sys.argv[1] == "generate_noise":
        from wakeworddetection.generate_noise import generate_silence
        generate_silence()
    elif sys.argv[1] == "train":
        from wakeworddetection.train import train_model
        train_model()
    elif sys.argv[1] == "convert":
        from wakeworddetection.convert_to_tflite import convert_to_tflite
        convert_to_tflite()
    elif sys.argv[1] == "test":
        from wakeworddetection.test import test_model
        test_model()