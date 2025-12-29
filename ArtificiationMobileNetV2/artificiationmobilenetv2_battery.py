import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.applications import MobileNetV2
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import RandomFlip, RandomRotation
import zipfile
from PIL import Image

from google.colab import drive
drive.mount('/content/drive')

#selects and unzips file
zip_file = '/content/drive/MyDrive/battery.zip'
#creates new directory where uncompressed file is extracted to
extract_to = "dataset_original"
os.makedirs(extract_to, exist_ok=True)

with zipfile.ZipFile(zip_file, 'r') as zf:
    zf.extractall(extract_to)

#category map that reorganizes the kaggle notebook folders into the 4 classes based on recology guidelines
CATEGORY_MAP = {
    "metal":"recycle",
    "glass":"recycle",
    "biological":"compost",
    "paper":"recycle",
    "battery":"toxic",
    "trash":"landfill",
    "cardboard":"recycle",
    "shoes":"landfill"}

SRC = "dataset_original"
DEST = "dataset"

#organizes the orgiginal data into the four classes based on category map
for cls in ["compost", "landfill", "recycle", "toxic"]:
    os.makedirs(os.path.join(DEST, cls), exist_ok=True)

for root, dirs, files in os.walk(SRC):
    folder = os.path.basename(root)

    if folder in CATEGORY_MAP:
        cls = CATEGORY_MAP[folder]
        print(f"Processing {folder} → {cls}")

        for file in files:
            if file.lower().endswith((".jpg")):
                src = os.path.join(root, file)
                dst = os.path.join(DEST, cls, file)

DEST1 = "split_dataset"
os.makedirs(DEST1, exist_ok=True)

#split train (70%), val (15%), test (15%)
for cls in ["compost", "landfill", "recycle", "toxic"]:
    images = os.listdir(os.path.join(DEST, cls))

    train_val, test_imgs = train_test_split(images, test_size=0.15, random_state=42)
    train_imgs, val_imgs = train_test_split(train_val, test_size=(0.15 / 0.85), random_state=42)

    for folder in ["train", "val", "test"]:
        os.makedirs(f"{DEST1}/{folder}/{cls}", exist_ok=True)
    #copies images into respective folder based on subset and class
    for img in train_imgs:
        shutil.copy(os.path.join(DEST, cls, img), f"{DEST1}/train/{cls}")
    for img in val_imgs:
        shutil.copy(os.path.join(DEST, cls, img), f"{DEST1}/val/{cls}")
    for img in test_imgs:
        shutil.copy(os.path.join(DEST, cls, img), f"{DEST1}/test/{cls}")

#searches for bad images
def find_bad_images(root):
    paths = glob.glob(os.path.join(root, "*", "*"))
    bad = []
    for p in paths:
        try:
            b = tf.io.read_file(p)
            _ = tf.image.decode_image(b, channels=3, expand_animations=False)
        except Exception as e:
            bad.append((p, str(e)))
            print("\nBAD FILE:", p)
    return bad

bad_train = find_bad_images(f"{DEST1}/train")
bad_val   = find_bad_images(f"{DEST1}/val")
bad_test  = find_bad_images(f"{DEST1}/test")

#removed bad files
for bad_list in [bad_train, bad_val, bad_test]:
    for p, _ in bad_list:
        os.remove(p)

#chose 32 as batch size because optimal for 10,000+ images
train_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DEST1}/train",
    image_size= (224, 224),
    batch_size=32)

val_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DEST1}/val",
    image_size= (224, 224),
    batch_size=32)

test_ds = tf.keras.utils.image_dataset_from_directory(
    f"{DEST1}/test",
    image_size= (224, 224),
    batch_size=32)

#to ensure consistency, created list of class names
class_names = train_ds.class_names

#autotuning makes training more efficient/faster
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

#essentially randomly augments data in each epoch to increase variation
data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1),])

#collects labels
y_train = np.concatenate([y.numpy() for _, y in train_ds])

#calls MobileNetV2 baseline model
base_model = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')

#freezes all layers except that last 40 to allow fine tuning
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

lr = 3e-5
inputs = tf.keras.Input(shape=(224,224,3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation=None)(x) #logits so no softmax activation needed
model = Model(inputs, outputs)

#create class weights using compute_class_weight from sklearn
from sklearn.utils.class_weight import compute_class_weight

classes = np.array([0, 1, 2, 3])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train)
class_weights_dict = dict(enumerate(weights))
print(class_weights_dict)

#define cross entropy loss
loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)

#compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr),
    loss= loss,
    metrics=['accuracy'])

#fit model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weights_dict_cap)

# path where model is saved in gDrive
model_path = '/content/drive/MyDrive/ArtificiationMobileNetV2_battery2.keras'

model.save(model_path)

#plot train vs val loss
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend(); plt.show()

#plot train vs val accuracy
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend(); plt.show()

#plot confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

y_true = np.concatenate([y.numpy() for _, y in test_ds])

# predict labels using argmax (model outputs)
y_pred = np.argmax(model.predict(test_ds), axis=1)

# defining confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

print(classification_report(y_true, y_pred, target_names=class_names))

import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()