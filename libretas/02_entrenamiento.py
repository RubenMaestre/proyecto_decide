# entrenamiento.py
# Este script procesa documentos de texto y sus entidades para crear archivos binarios
# que pueden ser utilizados para entrenar y validar un modelo de spaCy en español...
# Usamos plantillas generadas previamente y etiquetas para entrenar el modelo...

import os
import json
import spacy
from spacy.tokens import DocBin
import random

# Ruta a la carpeta de los textos extraídos y los JSON con las etiquetas...
data_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/datos/'

# Inicializar spaCy...
# Usar un modelo en blanco para español, que entrenaremos desde cero con nuestros datos.
nlp = spacy.blank('es')

# Recopilar y procesar todos los archivos JSON en el directorio de datos...
all_files = [file_name for file_name in os.listdir(data_folder) if file_name.endswith('.json')]

# Barajar los archivos para una distribución aleatoria...
random.shuffle(all_files)

# Dividir los archivos en conjuntos de entrenamiento y validación (80%-20%)...
split_index = int(len(all_files) * 0.8)
train_files = all_files[:split_index]
val_files = all_files[split_index:]

# Función para procesar archivos y crear DocBin...
# DocBin es una estructura de datos eficiente para almacenar múltiples objetos Doc, que representa nuestros documentos procesados con spaCy.
# Procesa cada archivo, crea el objeto Doc de spaCy y añade las entidades.
# Los documentos inválidos son descartados y se cuentan... así podemos evaluar si hay muchos errores en el procesamiento del entrenamiento.
def process_files(file_list, docbin):
    discarded_count = 0
    for file_name in file_list:
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        text = data['text']
        entities = data['entities']

        # Crear ejemplo de spaCy...
        doc = nlp.make_doc(text)
        spans = []
        valid = True
        for start, end, label in entities:
            if start < len(doc.text) and end <= len(doc.text):
                entity_text = text[start:end]
                # Verificar que el texto de la entidad coincida con lo que debería ser...
                if entity_text == text[start:end]:
                    span = doc.char_span(start, end, label=label)
                    if span is not None:
                        spans.append(span)
                    else:
                        valid = False
                        print(f"Entidad inválida: {label} ({start}-{end}) en el archivo {file_name}, texto: '{entity_text}'")
                        break
                else:
                    valid = False
                    print(f"Texto de entidad no coincide: {label} ({start}-{end}) en el archivo {file_name}, texto entidad: '{entity_text}', texto real: '{text[start:end]}'")
                    break
            else:
                valid = False
                print(f"Índice fuera de rango: {label} ({start}-{end}) en el archivo {file_name}, longitud del texto: {len(doc.text)}, texto: '{text[start:end]}'")
                break

        if valid:
            doc.ents = spans
            docbin.add(doc)
        else:
            discarded_count += 1
            print(f"Documento inválido descartado: {file_name}")

    return discarded_count

# Crear DocBin para entrenamiento y validación...
train_db = DocBin()
val_db = DocBin()

# Procesar archivos de entrenamiento...
train_discarded = process_files(train_files, train_db)
# Procesar archivos de validación...
val_discarded = process_files(val_files, val_db)

# Guardar los datos de entrenamiento y validación en formato binario de spaCy
train_output_path = 'train_data.spacy'
val_output_path = 'val_data.spacy'
train_db.to_disk(train_output_path)
val_db.to_disk(val_output_path)
print(f"Datos de entrenamiento guardados en '{train_output_path}'")
print(f"Datos de validación guardados en '{val_output_path}'")

# Imprimir resumen de archivos descartados. Así vemos si hay muchos errores a la hora del entrenamiento al montar los archivos del DocBin
print(f"Archivos descartados durante el entrenamiento: {train_discarded}")
print(f"Archivos descartados durante la validación: {val_discarded}")
