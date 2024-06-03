# 02_procesamiento.py
# Aquí lo que hacemos es preparar todo para realizar la tokenización que vamos a necesitar para BERT y ROBERTA.
# IMPORTANTE: Si no has ejecutado el 02_tokenizador.py tienes que comentar las partes del tokenizador personalizado para no ejecutarlas aquí.

import os
import json
import random
from collections import Counter
from tokenizers import Tokenizer
from transformers import RobertaTokenizerFast

# Cargar el tokenizador entrenado
tokenizer_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/tokenizer/"
tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
tokenizer = RobertaTokenizerFast(tokenizer_object=tokenizer, vocab_file=os.path.join(tokenizer_dir, "vocab.txt"))

# Directorio de entrada y directorio de salida
input_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/datos/'
output_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/datos_roberta/'

# Si el directorio de salida no existe, lo creamos
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función para leer los datos de los archivos JSON
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# Función para preparar los datos
def prepare_data(data):
    processed_data = []

    # Nos aseguramos de que data es un diccionario
    if not isinstance(data, dict):
        print(f"Expected data to be a dictionary, but got {type(data)}")
        return processed_data

    # Verificamos la existencia de claves 'text' y 'entities'
    if 'text' not in data or 'entities' not in data:
        print(f"Missing 'text' or 'entities' in data: {data}")
        return processed_data

    text = data['text']
    labels = ["O"] * len(text)  # Inicializamos todas las etiquetas como "O"
    for entity in data['entities']:
        start, end, label = entity
        if start >= len(text) or end > len(text):
            print(f"Entity {entity} fuera de rango para el texto de longitud {len(text)}")
            continue
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"

    # Realizamos la tokenización
    encoding = tokenizer(text, truncation=True, padding='max_length', return_offsets_mapping=True)
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])

    # Alineamos las etiquetas con los tokens
    word_ids = encoding.word_ids()
    aligned_labels = []
    previous_word_idx = None
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append("O")
        elif word_id != previous_word_idx:
            if text[word_id] == '|':
                aligned_labels.append("O")
            else:
                aligned_labels.append(labels[word_id])
        else:
            aligned_labels.append(labels[word_id] if labels[word_id].startswith("I-") else "O")
        previous_word_idx = word_id

    processed_data.append({
        "tokens": tokens,
        "attention_mask": encoding['attention_mask'],
        "token_type_ids": encoding.get('token_type_ids', [0]*len(encoding['input_ids'])),
        "labels": aligned_labels
    })

    return processed_data

# Función para reducir las etiquetas "O"
def reduce_O_labels(data, reduction_rate=0.75):
    for item in data:
        labels = item['labels']
        new_labels = []
        for label in labels:
            if label == "O" and random.random() > reduction_rate:
                new_labels.append(label)
            elif label != "O":
                new_labels.append(label)
        item['labels'] = new_labels
    return data

# Función para balancear los datos
def balance_data(data):
    labels = [item['labels'] for item in data]
    flat_labels = [label for sublist in labels for label in sublist]
    label_counts = Counter(flat_labels)
    
    min_label_count = min(label_counts.values())

    balanced_data = []
    for label in label_counts:
        label_data = [item for item in data if label in item['labels']]
        balanced_data.extend(label_data[:min_label_count])
    
    return balanced_data

# Procesar archivos
def process_files(input_dir, output_dir, num_files=500, max_records_per_file=500, reduction_rate=0.5):
    all_data = []
    file_count = 0

    # Obtenemos todos los archivos JSON en el directorio
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    # Seleccionamos un subconjunto aleatorio de archivos
    selected_files = random.sample(all_files, min(num_files, len(all_files)))

    for file_name in selected_files:
        file_path = os.path.join(input_dir, file_name)
        data = load_data(file_path)
        print(f"Processing file: {file_name}")
        prepared_data = prepare_data(data)
        all_data.extend(prepared_data)
    
    # Reducimos las etiquetas "O"
    all_data = reduce_O_labels(all_data, reduction_rate)
    
    # Balanceamos los datos
    all_data = balance_data(all_data)
    
    # Guardamos en un nuevo archivo cuando se alcance el límite de registros
    while len(all_data) >= max_records_per_file:
        output_file = os.path.join(output_dir, f'roberta_prepared_data_{file_count}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data[:max_records_per_file], f, ensure_ascii=False, indent=4)
        all_data = all_data[max_records_per_file:]  # Eliminamos los registros ya guardados
        file_count += 1

    # Guardamos cualquier dato restante en un archivo final
    if all_data:
        output_file = os.path.join(output_dir, f'roberta_prepared_data_{file_count}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    process_files(input_dir, output_dir, num_files=1000, max_records_per_file=500, reduction_rate=0.5)
    print(f"Data processed and saved to {output_dir}")
