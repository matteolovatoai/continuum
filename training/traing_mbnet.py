import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- 1. OTTIMIZZAZIONE PER APPLE SILICON (M2) ---
# Verifica se la GPU è visibile
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("✅ GPU Apple Silicon rilevata e attiva!")
else:
    print("⚠️ GPU non trovata, lo script userà la CPU (più lento).")

# --- 2. CONFIGURAZIONE PERCORSI ---
# Ci muoviamo correttamente tra le cartelle training -> data e models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
DATA_PATH = os.path.join(ROOT_DIR, "data", "classification")
SAVE_DIR = os.path.join(ROOT_DIR, "models")
SAVE_PATH = os.path.join(SAVE_DIR, "modello_lego_v1.h5")

# Parametri tecnici
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 3. CARICAMENTO DATI ---
print(f"Sto caricando le immagini da: {DATA_PATH}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Colori rilevati: {class_names}")

# --- 4. PIPELINE DI PERFORMANCE (MPS OPTIMIZED) ---
# Questo rende l'allenamento fluido sulla memoria unificata dell'M2
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 5. ARCHITETTURA DEL MODELLO ---
# Data augmentation leggera per gestire eventuali piccole rotazioni o riflessi
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
])

# Base model MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Congeliamo i pesi di ImageNet

# Assemblaggio finale
model = models.Sequential([
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1), # Preprocessing richiesto da MobileNetV2
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. ALLENAMENTO ---
print("\nInizio addestramento...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# --- 7. SALVATAGGIO ---
model.save(SAVE_PATH)
print(f"\n✅ Modello salvato con successo in: {SAVE_PATH}")