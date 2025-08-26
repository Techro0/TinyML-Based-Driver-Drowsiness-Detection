import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 64
BATCH = 64
TEST_DIR = Path("data/eyes_split/test")

def main():
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, labels="inferred", label_mode="int", color_mode="grayscale",
        batch_size=BATCH, image_size=(IMG_SIZE, IMG_SIZE), shuffle=False
    ).map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))

    class_names = ds.class_names
    model = tf.keras.models.load_model("models/best_model.keras")

    y_true = []
    y_pred = []
    for x,y in ds:
        y_true.extend(y.numpy().tolist())
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())

    print("Classes:", class_names)
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=class_names))

if __name__ == "__main__":
    main()