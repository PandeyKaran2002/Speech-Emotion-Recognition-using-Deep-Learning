import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import regularizers
from CutMix_Generator import CutMixGenerator

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
DATASET_DIR = "spectrograms"
CSV_PATH = "Col_features.csv"
PLOTS_DIR = "cutmix_multi-input_plots"
AUTOTUNE = tf.data.AUTOTUNE
os.makedirs(PLOTS_DIR, exist_ok=True)

csv_raw = pd.read_csv(CSV_PATH, header=None)
num_features = csv_raw.shape[1] - 2
feature_cols = [f'f{i}' for i in range(num_features)]
csv_raw.columns = feature_cols + ['filename', 'label']
csv_raw['filename'] = csv_raw['filename'].astype(str)

class_names = sorted(os.listdir(DATASET_DIR))
label_to_index = {name: i for i, name in enumerate(class_names)}
class_indices = label_to_index

scaler = StandardScaler()
scaled_feats = scaler.fit_transform(csv_raw[feature_cols])
csv_raw[feature_cols] = scaled_feats


feature_lookup = csv_raw.set_index('filename')[feature_cols]

def process_row(row, augment=False):
    path = row['filepath']
    label = row['label']
    filename = os.path.basename(path)
    csv_feats = feature_lookup.loc[filename].astype(np.float32).values
    label_idx = label_to_index[label]
    one_hot = tf.one_hot(label_idx, len(class_names))
    return path, one_hot, csv_feats

def load_image_and_features(path, label, feats, augment=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    if augment:
        
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)

        freq_mask = tf.random.uniform([], minval=0, maxval=16, dtype=tf.int32)
        image = tf.tensor_scatter_nd_update(image, indices=tf.reshape(tf.range(freq_mask), (-1, 1)), updates=tf.zeros_like(image[:freq_mask]))

        time_mask = tf.random.uniform([], minval=0, maxval=16, dtype=tf.int32)
        image = tf.transpose(image, perm=[1,0,2])  
        image = tf.tensor_scatter_nd_update(image, indices=tf.reshape(tf.range(time_mask), (-1, 1)), updates=tf.zeros_like(image[:time_mask]))
        image = tf.transpose(image, perm=[1,0,2])  

        sad_idx, happy_idx = class_indices['sad'], class_indices['happy']
        label_class = tf.argmax(label)
        is_sad_or_happy = tf.reduce_any(tf.equal(label_class, [sad_idx, happy_idx]))
        def extra_aug(img):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_saturation(img, 0.6, 1.4)
            return img
        image = tf.cond(is_sad_or_happy, lambda: extra_aug(image), lambda: image)

    return (image, feats), label

def make_dataset(df, augment=False):
    data = [process_row(row, augment) for _, row in df.iterrows()]
    paths, labels, feats = zip(*data)
    paths = tf.constant(paths)
    labels = tf.constant(np.stack(labels), dtype=tf.float32)
    feats = tf.constant(np.stack(feats), dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, feats))
    ds = ds.map(lambda p, l, f: load_image_and_features(p, l, f, augment), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

filepaths = []
labels = []
for emotion in os.listdir(DATASET_DIR):
    for file in os.listdir(os.path.join(DATASET_DIR, emotion)):
        filepaths.append(os.path.join(DATASET_DIR, emotion, file))
        labels.append(emotion)

df = pd.DataFrame({'filepath': filepaths, 'label': labels})
df = df.sample(frac=1, random_state=42)
split = int(0.8 * len(df))
train_df, val_df = df[:split], df[split:]
train_ds = make_dataset(train_df, augment=False)

cutmix_train_ds = CutMixGenerator(
    train_ds,
    alpha=1.0).get_generator()

val_ds = make_dataset(val_df, augment=False)

cutmix_train_ds = cutmix_train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)


y_train_labels = [label_to_index[label] for label in train_df['label']]
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights = dict(enumerate(class_weights))

image_input = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='image_input')
csv_input = layers.Input(shape=(num_features,), name='csv_input')

x = layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(image_input)
x = layers.MaxPooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
x = layers.MaxPooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Conv2D(256, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
x = layers.MaxPooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(x)
x = layers.Dropout(0.3)(x)

y = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(csv_input)
y = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-3))(y)

z = layers.concatenate([x, y])
output = layers.Dense(len(class_names), activation='softmax')(z)

model = models.Model(inputs=[image_input, csv_input], outputs=output)
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.3), metrics=['accuracy'])

model.summary()

class PerClassAccuracyCallback(callbacks.Callback):
    def __init__(self, val_dataset, class_names):
        super().__init__()
        self.val_dataset = val_dataset
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = [], []
        for (images, feats), labels in self.val_dataset:
            preds = self.model.predict([images, feats], verbose=0)
            y_true.append(tf.argmax(labels, axis=1).numpy())
            y_pred.append(tf.argmax(preds, axis=1).numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        per_class = cm.diagonal() / cm.sum(axis=1)
        print(f"\n📊 Per-class Validation Accuracy after Epoch {epoch+1}:")
        for i, acc in enumerate(per_class):
            print(f"  {self.class_names[i]}: {acc:.2f}")

per_class_cb = PerClassAccuracyCallback(val_ds, class_names)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint = callbacks.ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True)

history = model.fit(cutmix_train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[per_class_cb, reduce_lr, checkpoint], class_weight=class_weights, verbose=1)

y_true, y_pred = [], []
for (images, feats), labels in val_ds:
    preds = model.predict([images, feats])
    y_true.append(np.argmax(labels, axis=1))
    y_pred.append(np.argmax(preds, axis=1))
y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(norm_cm, annot=True, xticklabels=class_names, yticklabels=class_names, fmt=".2f", cmap='Blues')
plt.title("Normalized Confusion Matrix")
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], label="Train Acc")
plt.plot(history.history['val_accuracy'], label="Val Acc")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(os.path.join(PLOTS_DIR, "train_val_accuracy.png"))
plt.show()

per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(10, 6))
bars = plt.bar(class_names, per_class_accuracy, color='skyblue')
plt.ylim(0, 1)
plt.title("Per-Class Validation Accuracy")
for bar, acc in zip(bars, per_class_accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2.0, acc + 0.02, f"{acc:.2f}", ha='center')
plt.grid(axis='y')
plt.savefig(os.path.join(PLOTS_DIR, "per_class_accuracy.png"))
plt.show()

print(f"\n✅ Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"✅ Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
