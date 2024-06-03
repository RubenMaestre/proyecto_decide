# 06_metricas.py
# Este script lo hemos hecho para saber el % que tienen cada una de las categorías dentro del peso de los datos que hemos entrenado
# para saber cuánto de desbalanceados están nuestros datos.

import os
import json
import torch
from collections import Counter
from transformers import RobertaTokenizerFast

# Mapeo de etiquetas a índices
label2id = {
    "O": 0,
    "B-nombre_cliente": 1, "I-nombre_cliente": 2,
    "B-dni_cliente": 3, "I-dni_cliente": 4,
    "B-calle_cliente": 5, "I-calle_cliente": 6,
    "B-cp_cliente": 7, "I-cp_cliente": 8,
    "B-poblacion_cliente": 9, "I-poblacion_cliente": 10,
    "B-provincia_cliente": 11, "I-provincia_cliente": 12,
    "B-nombre_comercializadora": 13, "I-nombre_comercializadora": 14,
    "B-cif_comercializadora": 15, "I-cif_comercializadora": 16,
    "B-direccion_comercializadora": 17, "I-direccion_comercializadora": 18,
    "B-cp_comercializadora": 19, "I-cp_comercializadora": 20,
    "B-poblacion_comercializadora": 21, "I-poblacion_comercializadora": 22,
    "B-provincia_comercializadora": 23, "I-provincia_comercializadora": 24,
    "B-numero_factura": 25, "I-numero_factura": 26,
    "B-inicio_periodo": 27, "I-inicio_periodo": 28,
    "B-fin_periodo": 29, "I-fin_periodo": 30,
    "B-importe_factura": 31, "I-importe_factura": 32,
    "B-fecha_cargo": 33, "I-fecha_cargo": 34,
    "B-consumo_periodo": 35, "I-consumo_periodo": 36,
    "B-potencia_contratada": 37, "I-potencia_contratada": 38,
}

# Función para cargar los datos preprocesados
def load_preprocessed_data(data_dir, tokenizer):
    encodings = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    labels = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    input_ids = tokenizer.convert_tokens_to_ids(item['tokens'])
                    encodings['input_ids'].append(input_ids)
                    encodings['attention_mask'].append(item['attention_mask'])
                    encodings['token_type_ids'].append(item.get('token_type_ids', [0] * len(input_ids)))
                    labels.append([label2id[label] for label in item['labels']])

    # Aplicar padding a las secuencias de tokens y etiquetas
    max_len = max(len(seq) for seq in encodings['input_ids'])
    encodings = {key: [seq + [0] * (max_len - len(seq)) for seq in val] for key, val in encodings.items()}
    labels = [seq + [-100] * (max_len - len(seq)) for seq in labels]

    # Convertir las listas a tensores de PyTorch
    encodings = {key: torch.tensor(val) for key, val in encodings.items()}
    labels = torch.tensor(labels)

    return encodings, labels

# Función para calcular los porcentajes de cada etiqueta
def calculate_label_percentages(labels):
    label_counts = Counter(labels.flatten().tolist())
    total_labels = sum(label_counts.values())
    label_percentages = {label: count / total_labels * 100 for label, count in label_counts.items()}
    return label_percentages

# Cargar el tokenizer
data_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/datos_roberta/'
tokenizer = RobertaTokenizerFast.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')

# Cargar los datos preprocesados
encodings, labels = load_preprocessed_data(data_dir, tokenizer)

# Calcular los porcentajes de cada etiqueta
label_percentages = calculate_label_percentages(labels)

# Imprimir los porcentajes
for label, percentage in label_percentages.items():
    label_name = [k for k, v in label2id.items() if v == label][0]
    print(f"{label_name}: {percentage:.2f}%")
