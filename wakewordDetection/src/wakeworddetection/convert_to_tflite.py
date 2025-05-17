import tensorflow as tf

MODEL_PATH = './processed/wakeword_model.keras'
OUTPUT_PATH = './processed/wakeword_model.tflite'

def convert_to_tflite():
    model = tf.keras.models.load_model(MODEL_PATH)
   
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Enable optimizations for size and performance
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)

    print("Converted to TFLite!")