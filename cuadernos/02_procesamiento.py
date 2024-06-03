# 02_procesamiento.py
# En este script vamos a procesar los textos extraídos y los JSON con las etiquetas para generar los conjuntos de datos de entrenamiento y validación para spaCy.

import os
import json
import spacy
from spacy.tokens import DocBin, Span
from spacy.training import Example
import random

# Ruta a la carpeta de los textos extraídos y los JSON con las etiquetas
text_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/cuadernos/pre'
json_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/training'

# Lista de categorías que estamos buscando
categories = [
    'nombre_cliente', 'dni_cliente', 'calle_cliente', 'cp_cliente', 'población_cliente', 'provincia_cliente',
    'nombre_comercializadora', 'cif_comercializadora', 'dirección_comercializadora', 'cp_comercializadora',
    'población_comercializadora', 'provincia_comercializadora', 'número_factura', 'inicio_periodo', 'fin_periodo',
    'importe_factura', 'fecha_cargo', 'consumo_periodo', 'potencia_contratada'
]

# Función para cargar y limpiar texto
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Función para cargar JSON
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Función para verificar superposiciones
def check_overlap(new_entity, existing_entities):
    new_start, new_end, _ = new_entity
    for start, end, _ in existing_entities:
        if (new_start < end and new_end > start):
            return True
    return False

# Inicializar spaCy
nlp = spacy.blank('es')  # Usar un modelo en blanco para español

# Listas para almacenar ejemplos de entrenamiento y validación
train_docs = []
val_docs = []

# Recopilar y procesar todos los archivos
all_files = [file_name for file_name in os.listdir(text_folder) if file_name.endswith('.txt')]

# Barajar los archivos para una distribución aleatoria
random.shuffle(all_files)

# Dividir los archivos en entrenamiento y validación (80%-20%)
split_index = int(len(all_files) * 0.8)
train_files = all_files[:split_index]
val_files = all_files[split_index:]

# Función para procesar archivos y crear DocBin
def process_files(file_list, docbin):
    for file_name in file_list:
        text_path = os.path.join(text_folder, file_name)
        json_path = os.path.join(json_folder, file_name.replace('.txt', '.json'))

        if not os.path.exists(json_path):
            continue

        text = load_text(text_path)
        entities = load_json(json_path)

        # Crear anotaciones de entidades
        annotations = []
        for label, value in entities.items():
            if label in categories:
                value_str = str(value).lower()  # Convertir el valor a cadena y a minúsculas
                start = text.find(value_str)
                if start != -1:
                    end = start + len(value_str)
                    new_entity = (start, end, label)
                    if not check_overlap(new_entity, annotations):
                        annotations.append(new_entity)

        # Crear ejemplo de spaCy
        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in annotations:
            # Asegurarse de que los índices de las entidades estén dentro del rango del texto
            if start < len(doc) and end <= len(doc):
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    spans.append(span)

        doc.ents = spans
        docbin.add(doc)

# Crear DocBin para entrenamiento y validación
train_db = DocBin()
val_db = DocBin()

# Procesar archivos de entrenamiento
process_files(train_files, train_db)
# Procesar archivos de validación
process_files(val_files, val_db)

# Guardar los datos de entrenamiento y validación en formato binario de spaCy
train_output_path = 'train_data.spacy'
val_output_path = 'val_data.spacy'
train_db.to_disk(train_output_path)
val_db.to_disk(val_output_path)
print(f"Datos de entrenamiento guardados en '{train_output_path}'")
print(f"Datos de validación guardados en '{val_output_path}'")
