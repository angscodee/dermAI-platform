# -*- coding: utf-8 -*-
"""
entrenamiento_colab.py
Script de entrenamiento completo para Google Colab para entrenar los modelos de DermAI
en el dataset ISIC 2019 (clasificación binaria benigno/maligno).

Este script descarga el dataset desde Kaggle, procesa las etiquetas,
aplica data augmentation, entrena las 5 arquitecturas (EfficientNetB0,
ResNet50, MobileNetV2, DenseNet121, InceptionV3) y guarda los modelos entrenados.
"""

import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración global
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30  # Ajustable, con EarlyStopping se detendrá antes si converge
LEARNING_RATE = 1e-4

# Nombres de los modelos y sus correspondientes clases base de Keras
MODEL_ARCHITECTURES = {
    "EfficientNetB0": keras.applications.EfficientNetB0,
    "ResNet50": keras.applications.ResNet50,
    "MobileNetV2": keras.applications.MobileNetV2,
    "DenseNet121": keras.applications.DenseNet121,
    "InceptionV3": keras.applications.InceptionV3,
}

def setup_kaggle():
    """Configura la API de Kaggle en Colab utilizando el token o el archivo json."""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Token provisto por la nueva API de Kaggle
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if not token:
        # Token ingresado directamente
        token = "KGAT_4a8b25ff40caeda1ef26d9ebd1938db4"
        os.environ["KAGGLE_API_TOKEN"] = token

    if token:
        access_token_path = os.path.join(kaggle_dir, "access_token")
        with open(access_token_path, "w") as f:
            f.write(token)
        os.chmod(access_token_path, 0o600)
        print("✅ API Token de Kaggle configurado con éxito.")
        return

    if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
        print("\n🔑 CONFIGURACIÓN DE KAGGLE")
        print("Por favor, sube tu archivo 'kaggle.json' obtenido de Kaggle (My Account -> Create New API Token)")
        from google.colab import files
        uploaded = files.upload()
        if 'kaggle.json' in uploaded:
            with open(os.path.join(kaggle_dir, 'kaggle.json'), 'wb') as f:
                f.write(uploaded['kaggle.json'])
            os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
            print("✅ API Token de Kaggle configurado correctamente.")
        else:
            raise FileNotFoundError("No se configuró el token de Kaggle.")

def download_dataset():
    """Descarga y extrae el dataset ISIC 2019 desde Kaggle."""
    dataset_name = "andrewmvd/isic-2019"
    zip_path = "isic-2019.zip"
    
    if not os.path.exists("ISIC_2019_Training_GroundTruth.csv"):
        print(f"\n⏳ Descargando dataset {dataset_name} de Kaggle...")
        os.system(f"kaggle datasets download -d {dataset_name} -p .")
        
        # Buscar el archivo zip descargado (a veces varía el nombre exacto)
        for file in os.listdir("."):
            if file.endswith(".zip") and "isic" in file.lower():
                zip_path = file
                break
                
        print(f"📦 Extrayendo {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("✅ Dataset extraído con éxito.")
    else:
        print("✅ El dataset ISIC 2019 ya se encuentra descargado y descomprimido.")

def prepare_data_labels():
    """
    Lee las etiquetas de GroundTruth, genera la columna binaria benigno (0) / maligno (1)
    y limpia las rutas de las imágenes para apuntar al directorio correcto.
    """
    csv_path = "ISIC_2019_Training_GroundTruth.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo de etiquetas {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Mapear clases de ISIC 2019 a clasificación binaria
    # Maligno = MEL (Melanoma), BCC (Basal Cell Carcinoma), AK (Actinic Keratosis), SCC (Squamous Cell Carcinoma)
    # Benigno = NV (Nevus), BKL (Benign Keratosis), DF (Dermatofibroma), VASC (Vascular Lesion), UNK (Unknown)
    malignant_cols = ['MEL', 'BCC', 'AK', 'SCC']
    df['is_malignant'] = df[malignant_cols].max(axis=1).astype(int)
    
    # Crear columna con ruta de imagen (añadiendo extensión .jpg)
    # Dependiendo de cómo se extraiga el zip, las imágenes pueden estar en:
    # ISIC_2019_Training_Input/ISIC_2019_Training_Input/ o ISIC_2019_Training_Input/
    img_dir_candidates = [
        "ISIC_2019_Training_Input/ISIC_2019_Training_Input",
        "ISIC_2019_Training_Input",
        "isic-2019/ISIC_2019_Training_Input/ISIC_2019_Training_Input"
    ]
    
    base_img_dir = None
    for candidate in img_dir_candidates:
        if os.path.exists(candidate):
            base_img_dir = candidate
            break
            
    if base_img_dir is None:
        raise FileNotFoundError("No se encontró el directorio de imágenes ISIC_2019_Training_Input")
        
    print(f"📂 Carpeta de imágenes seleccionada: {base_img_dir}")
    
    df['image_path'] = df['image'].apply(lambda x: os.path.join(base_img_dir, f"{x}.jpg"))
    
    # Verificar que las imágenes existan físicamente (algunas descargas de prueba o corruptas pueden fallar)
    print("⏳ Verificando existencia física de imágenes...")
    df['exists'] = df['image_path'].apply(os.path.exists)
    df = df[df['exists'] == True].reset_index(drop=True)
    print(f"✅ Total de imágenes válidas encontradas: {len(df)}")
    
    # Convertir etiqueta a string para Keras flow_from_dataframe (modo binary/categorical)
    df['label_str'] = df['is_malignant'].astype(str)
    
    return df

def build_model(model_name, base_arch_func):
    """
    Construye un modelo de Transfer Learning con la base seleccionada,
    un GlobalAveragePooling2D y una capa de salida densa con activación sigmoide.
    """
    print(f"\n🏗️ Construyendo arquitectura para {model_name}...")
    
    # Cargar el modelo base con pesos pre-entrenados de ImageNet
    base_model = base_arch_func(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    
    # Congelar el modelo base inicialmente (o dejarlo descongelado con lr bajo para fine-tuning)
    # Para el entrenamiento oficial, es mejor descongelar las últimas capas.
    # Aquí descongelamos las últimas 20 capas del modelo base para un fine-tuning fino
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
        
    # Construcción de la cabeza del clasificador
    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    
    # Preprocesamiento específico para cada arquitectura (algunas requieren escala de -1 a 1, otras 0 a 255)
    # NOTA: En la app usamos normalización general /255.0, por lo que adaptamos
    # la entrada del modelo para ser compatible con el rango [0, 1].
    x = inputs
    if "EfficientNet" in model_name:
        # EfficientNet tiene su propio escalado interno si se usa Keras Applications
        pass
    elif "ResNet" in model_name:
        # ResNet espera [0, 255] o preprocesamiento ResNet
        x = layers.Lambda(lambda v: v * 255.0)(x)
        x = keras.applications.resnet50.preprocess_input(x)
    elif "Inception" in model_name:
        # Inception espera [-1, 1]
        x = layers.Lambda(lambda v: (v - 0.5) * 2.0)(x)
    elif "DenseNet" in model_name:
        # DenseNet espera preprocesamiento específico
        x = layers.Lambda(lambda v: v * 255.0)(x)
        x = keras.applications.densenet.preprocess_input(x)
    elif "MobileNet" in model_name:
        # MobileNet espera [-1, 1]
        x = layers.Lambda(lambda v: (v - 0.5) * 2.0)(x)
        
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compilar con Adam y learning rate bajo para fine-tuning
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    return model

def calculate_metrics(y_true, y_pred_probs, threshold=0.5):
    """Calcula todas las métricas avanzadas (sensibilidad, especificidad, MCC, etc.)."""
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
        "mcc": mcc,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

def main():
    print("🚀 Iniciando Pipeline de Entrenamiento de DermAI para Google Colab")
    
    # 1. Configurar Kaggle y descargar dataset
    try:
        setup_kaggle()
    except Exception as e:
        print(f"⚠️ No se configuró Kaggle mediante API: {e}")
        print("Si ya subiste los archivos manualmente a Colab, continuaremos.")
        
    download_dataset()
    
    # 2. Cargar y procesar etiquetas
    df = prepare_data_labels()
    
    # Split de datos: 70% train / 10% val / 20% test
    # Usamos stratify para mantener la misma distribución de benigno/maligno en los splits
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['is_malignant']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=2/3, random_state=42, stratify=temp_df['is_malignant']
    )
    
    print(f"\n📊 Distribución de Datos:")
    print(f"  • Entrenamiento: {len(train_df)} imágenes")
    print(f"  • Validación:   {len(val_df)} imágenes")
    print(f"  • Test:         {len(test_df)} imágenes")
    
    # 3. Flujo de datos e imágenes (ImageDataGenerator)
    # Aplicar Data Augmentation para evitar sobreentrenamiento
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalización básica de píxeles a [0, 1]
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    
    # Validación y Test no deben tener Data Augmentation (solo reescalado)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Generadores
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image_path',
        y_col='label_str',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_dataframe(
        dataframe=val_df,
        x_col='image_path',
        y_col='label_str',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col='image_path',
        y_col='label_str',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    # Carpeta de salida para guardar modelos
    output_dir = "models_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Estructura para almacenar las métricas finales de todos los modelos entrenados
    resultados_totales = {}
    
    # 4. Entrenar y evaluar cada modelo
    for model_name, arch_func in MODEL_ARCHITECTURES.items():
        print(f"\n========================================================")
        print(f"🔥 COMENZANDO ENTRENAMIENTO DE: {model_name}")
        print(f"========================================================")
        
        model = build_model(model_name, arch_func)
        
        # Callbacks para un entrenamiento óptimo
        model_save_path = os.path.join(output_dir, f"{model_name.lower()}_isic2019.keras")
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=4,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Entrenamiento
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✅ Entrenamiento de {model_name} finalizado.")
        print(f"📦 Modelo guardado en: {model_save_path}")
        
        # Evaluar en el conjunto de test
        print(f"⏳ Evaluando en el conjunto de test...")
        test_generator.reset()
        predictions = model.predict(test_generator, verbose=1).flatten()
        y_true = test_generator.classes
        
        # Calcular y guardar métricas
        metrics = calculate_metrics(y_true, predictions)
        resultados_totales[model_name] = metrics
        
        print(f"\n📈 Métricas finales para {model_name} en Test:")
        print(f"  • Accuracy:    {metrics['accuracy']*100:.2f}%")
        print(f"  • Sensibilidad: {metrics['sensitivity']*100:.2f}%")
        print(f"  • Especificidad:{metrics['specificity']*100:.2f}%")
        print(f"  • MCC:          {metrics['mcc']:.4f}")
        print(f"  • Matriz de Confusión: {metrics['confusion_matrix']}")
        
    # 5. Generar reporte consolidado e instrucciones de exportación
    print("\n\n========================================================")
    print("🏁 PIPELINE COMPLETADO EXITOSAMENTE")
    print("========================================================")
    
    # Crear diccionario para st.session_state o config.py
    print("\n📋 Puedes copiar el siguiente diccionario y reemplazarlo en config.py:")
    print("----------------------------------------------------------------------")
    print("REAL_TRAINING_METRICS = {")
    for m_name, metrics in resultados_totales.items():
        print(f"    '{m_name}': {{")
        print(f"        'accuracy': {metrics['accuracy']:.4f},")
        print(f"        'sensitivity': {metrics['sensitivity']:.4f},")
        print(f"        'specificity': {metrics['specificity']:.4f},")
        print(f"        'precision': {metrics['precision']:.4f},")
        print(f"        'f1_score': {metrics['f1_score']:.4f},")
        print(f"        'mcc': {metrics['mcc']:.4f},")
        print(f"        'confusion_matrix': {metrics['confusion_matrix']},")
        print(f"    }},")
    print("}")
    print("----------------------------------------------------------------------")
    
    print("\n📦 ARCHIVOS LISTOS PARA DESCARGAR:")
    print("Los modelos se guardaron en la carpeta 'models_output/'.")
    print("Descárgalos y colócalos en la carpeta 'app/models/' de tu aplicación DermAI local:")
    for file in os.listdir(output_dir):
        print(f"  • {os.path.join(output_dir, file)}")

if __name__ == "__main__":
    # Evitar ejecutar si estamos importando desde Jupyter para control fino
    import sys
    if 'google.colab' in sys.modules or __name__ == "__main__":
        main()
