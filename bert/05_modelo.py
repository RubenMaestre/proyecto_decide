# 05_modelo.py
# Este script es para probar el modelo. Vamos a cargar el modelo entrenado y el tokenizador para realizar inferencias sobre textos nuevos.
# En este caso, usamos ROBERTA con un tokenizador personalizado, pero se puede adaptar para usar BERT u otros modelos y tokenizadores.

import os
import json
import torch
from transformers import RobertaTokenizerFast, RobertaForTokenClassification
from torch.nn import functional as F

# Mapeo de etiquetas a índices y viceversa
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
id2label = {v: k for k, v in label2id.items()}

# Categorías a extraer
categories = [
    "nombre_cliente", "dni_cliente", "calle_cliente", "cp_cliente", "poblacion_cliente", "provincia_cliente",
    "nombre_comercializadora", "cif_comercializadora", "direccion_comercializadora", "cp_comercializadora",
    "poblacion_comercializadora", "provincia_comercializadora", "numero_factura", "inicio_periodo", "fin_periodo",
    "importe_factura", "fecha_cargo", "consumo_periodo", "potencia_contratada"
]

# Cargar el modelo y el tokenizer
model_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/archivos_roberta_2/'
tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
model = RobertaForTokenClassification.from_pretrained(model_dir)

# Función para realizar la inferencia
def infer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
    
    entities = {}
    current_entity = None
    current_label = None

    for token, label_id in zip(tokens, predictions):
        label = id2label[label_id]
        if label == 'O':
            if current_entity and current_label:
                entities[current_label] = ' '.join(current_entity)
                current_entity = None
                current_label = None
        elif label.startswith('B-'):
            if current_entity and current_label:
                entities[current_label] = ' '.join(current_entity)
            current_entity = [token]
            current_label = label[2:]
        elif label.startswith('I-') and current_label == label[2:]:
            current_entity.append(token)

    if current_entity and current_label:
        entities[current_label] = ' '.join(current_entity)
    
    return entities

# Leer archivos de texto y realizar inferencia
input_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/facturas/'
output_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/validaciones/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for file_name in os.listdir(input_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        entities = infer(text)
        
        # Completar las categorías faltantes con valores vacíos
        for category in categories:
            if category not in entities:
                entities[category] = ""

        # Guardar el resultado en un archivo JSON
        output_file_path = os.path.join(output_dir, file_name.replace('.txt', '.json'))
        with open(output_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(entities, json_file, ensure_ascii=False, indent=4)

print("Inferencia completada y archivos JSON guardados.")
