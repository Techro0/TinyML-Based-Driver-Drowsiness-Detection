import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, callbacks
from pathlib import Path

IMG_SIZE = 64
BATCH = 64
EPOCHS = 20
SEED = 42

DATA_DIR = Path("data/eyes_split")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"

def build_tiny_ds_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 1), num_classes=2):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    def ds_block(x, filters, stride=1):
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x

    x = ds_block(x, 24, stride=2)   # 32x32
    x = ds_block(x, 32, stride=2)   # 16x16
    x = ds_block(x, 48, stride=2)   # 8x8
    x = ds_block(x, 64, stride=2)   # 4x4

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    model = models.Model(inp, out)
    return model

def make_ds(split_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=BATCH,
        image_size=(IMG_SIZE, IMG_SIZE),
        seed=SEED,
        shuffle=True
    )
    return ds

def main():
    train_ds = make_ds(TRAIN_DIR)
    val_ds   = make_ds(VAL_DIR)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)

    aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.05),
    ])

    def add_preprocess(ds):
        return ds.map(lambda x,y: (tf.cast(x, tf.float32)/255.0, y))

    train_ds = train_ds.map(lambda x,y: (aug(tf.cast(x, tf.float32)/255.0), y), num_parallel_calls=AUTOTUNE)
    val_ds   = add_preprocess(val_ds)

    model = build_tiny_ds_cnn()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    ckpt_dir = Path("models")
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=str(ckpt_dir / "best_model.keras"),
        save_best_only=True, monitor="val_accuracy", mode="max"
    )
    es_cb = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_accuracy", mode="max")

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[ckpt_cb, es_cb])
    model.save("models/saved_model")
    print("Training complete. Saved best model to models/best_model.keras and models/saved_model")

if __name__ == "__main__":
    main()