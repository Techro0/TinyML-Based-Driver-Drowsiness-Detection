import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image
import random

IMG_SIZE = 64
REP_SAMPLES = 200
TRAIN_DIR = Path("data/eyes_split/train")

def rep_data_gen():
    imgs = []
    for cls in ["open","close"]:
        imgs += list((TRAIN_DIR/cls).glob("*"))
    random.shuffle(imgs)
    for p in imgs[:REP_SAMPLES]:
        img = Image.open(p).convert("L").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))
        yield [arr]

def main():
    model = tf.keras.models.load_model("models/best_model.keras")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out = Path("models/eye_state_int8.tflite")
    out.write_bytes(tflite_model)
    Path("models/label_map.txt").write_text("0 open\n1 close\n")
    print(f"Saved {out}")

if __name__ == "__main__":
    main()