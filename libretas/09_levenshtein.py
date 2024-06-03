# levenshtein.py
# Este script se utiliza para calcular el score de precisión de nuestro modelo de extracción de entidades (NER).
# La métrica utilizada es la distancia de Levenshtein, que mide la similitud entre dos cadenas de texto.
# Nos enviarán estos archivos JSON que hemos obtenido, y ellos usarán un script para calcular el score
# que hemos obtenido. Este score se expresará en porcentaje y se calcula como la media de una métrica basada
# en la distancia de Levenshtein de todos los campos de todos los documentos.

import os
import json
import Levenshtein
from dateutil.parser import parse

# Directorios
original_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/training/'
extracted_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/validaciones/'

# Categorías de entidades
categories = [
    "nombre_cliente", "dni_cliente", "calle_cliente", "cp_cliente", "población_cliente",
    "provincia_cliente", "nombre_comercializadora", "cif_comercializadora", "dirección_comercializadora",
    "cp_comercializadora", "población_comercializadora", "provincia_comercializadora", "número_factura",
    "inicio_periodo", "fin_periodo", "importe_factura", "fecha_cargo", "consumo_periodo", "potencia_contratada"
]

# Función para cargar JSON
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Función para normalizar fechas al formato dd.mm.yyyy
def normalize_date(date_str):
    try:
        date_obj = parse(date_str, dayfirst=True)
        return date_obj.strftime("%d.%m.%Y")
    except (ValueError, OverflowError):
        return date_str

# Función para calcular el score basado en la distancia de Levenshtein
# La métrica utilizada es:
# Score = Σ (1 - L(ŝ, s) / len(s)) / n
# Donde L(a,b) es la distancia de Levenshtein entre las cadenas a y b, s es el string del campo i-ésimo,
# ŝ es el string de nuestra predicción para el campo i-ésimo, y len() devuelve la longitud de un string.
def calculate_levenshtein_score(original, extracted):
    score_sum = 0
    n = len(categories)
    
    for category in categories:
        original_value = str(original.get(category, "")).strip()
        extracted_value = str(extracted.get(category, "")).strip()
        
        # Normalizar fechas si el campo es una fecha
        if "fecha" in category or "periodo" in category:
            original_value = normalize_date(original_value)
            extracted_value = normalize_date(extracted_value)
        
        levenshtein_distance = Levenshtein.distance(original_value, extracted_value)
        max_len = max(len(original_value), 1)
        score_sum += (1 - (levenshtein_distance / max_len))
    
    return score_sum / n

# Inicializar la suma de scores
total_score = 0
num_files = 0

# Procesar cada archivo en el directorio de originales y extraídos
for filename in os.listdir(original_dir):
    if filename.endswith('.json'):
        original_file_path = os.path.join(original_dir, filename)
        extracted_file_path = os.path.join(extracted_dir, filename.replace('.json', '_result.json'))
        
        if os.path.exists(extracted_file_path):
            original_data = load_json(original_file_path)
            extracted_data = load_json(extracted_file_path)
            
            file_score = calculate_levenshtein_score(original_data, extracted_data)
            total_score += file_score
            num_files += 1

# Calcular y mostrar el score promedio
if num_files > 0:
    average_score = (total_score / num_files) * 100
    print(f"Score promedio: {average_score:.2f}%")
else:
    print("No hay archivos para calcular el score")
