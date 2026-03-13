import tensorflow as tf
import numpy as np
import os

# --- CONFIGURAZIONE ---
MODEL_PATH = "models/modello_lego_v1.h5"
# L'ordine deve essere alfabetico (lo stesso usato da train_ds.class_names)
class_names = ['bianco', 'blu', 'giallo', 'rosso'] 

# Caricamento del modello
print("Caricamento del modello in corso...")
model = tf.keras.models.load_model(MODEL_PATH)

def predict_lego(image_path):
    if not os.path.exists(image_path):
        print(f"Errore: File {image_path} non trovato.")
        return

    # 1. Caricamento e ridimensionamento (come nel training)
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    
    # 2. Conversione in array e aggiunta della dimensione "batch"
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Da (224,224,3) a (1,224,224,3)

    # 3. Predizione
    predictions = model.predict(img_array, verbose=0)
    
    # Se hai usato Softmax nell'ultimo layer, predictions[0] contiene le probabilità
    score = predictions[0]
    result_index = np.argmax(score)
    
    print("-" * 30)
    print(f"RISULTATO PER: {os.path.basename(image_path)}")
    print(f"Colore predetto: {class_names[result_index].upper()}")
    print(f"Confidenza: {100 * score[result_index]:.2f}%")
    print("-" * 30)

# --- PROVA PRATICA ---
# Metti qui il percorso di una foto che non era nel dataset!
#predict_lego("data/img_lego.jpg")