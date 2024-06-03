# medicion.py
# Este script se utiliza para medir la precisión de nuestro modelo de extracción de entidades (NER).
# Compara los resultados extraídos por el modelo con los datos originales y calcula la tasa de acierto
# para cada categoría de entidad y una tasa de acierto global...

import os
import json
from dateutil.parser import parse

# Directorios
original_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/training/'
extracted_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/validaciones/'

# Categorías de entidades
# Estas son las etiquetas que esperamos que el modelo haya reconocido correctamente en los textos.
categories = [
    "nombre_cliente", "dni_cliente", "calle_cliente", "cp_cliente", "población_cliente",
    "provincia_cliente", "nombre_comercializadora", "cif_comercializadora", "dirección_comercializadora",
    "cp_comercializadora", "población_comercializadora", "provincia_comercializadora", "número_factura",
    "inicio_periodo", "fin_periodo", "importe_factura", "fecha_cargo", "consumo_periodo", "potencia_contratada"
]

# Función para cargar JSON
# Cargamos los datos de un archivo JSON y los devolvemos como un diccionario.
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

# Función para comparar dos diccionarios y calcular la tasa de acierto
# Compara los valores de las entidades en los diccionarios original y extraído, y cuenta los aciertos.
def compare_dicts(original, extracted):
    total = 0
    correct = 0
    for category in categories:
        total += 1
        original_value = str(original.get(category, "")).strip()
        extracted_value = str(extracted.get(category, "")).strip()
        
        # Normalizar fechas si el campo es una fecha
        if "fecha" in category or "periodo" in category:
            original_value = normalize_date(original_value)
            extracted_value = normalize_date(extracted_value)
        
        if original_value == extracted_value:
            correct += 1
    return correct, total

# Inicializar contadores
# Estos contadores llevarán un registro de los aciertos por categoría y globalmente.
total_correct = {category: 0 for category in categories}
total_entries = {category: 0 for category in categories}
global_correct = 0
global_total = 0
document_accuracies = []

# Procesar cada archivo en el directorio de originales y extraídos
# Iteramos sobre cada archivo JSON en el directorio original y comparamos con el archivo correspondiente en el directorio extraído.
for filename in os.listdir(original_dir):
    if filename.endswith('.json'):
        original_file_path = os.path.join(original_dir, filename)
        extracted_file_path = os.path.join(extracted_dir, filename.replace('.json', '_result.json'))
        
        if os.path.exists(extracted_file_path):
            original_data = load_json(original_file_path)
            extracted_data = load_json(extracted_file_path)
            
            correct, total = compare_dicts(original_data, extracted_data)
            global_correct += correct
            global_total += total
            document_accuracies.append(correct / total if total > 0 else 0)

            for category in categories:
                total_entries[category] += 1
                original_value = str(original_data.get(category, "")).strip()
                extracted_value = str(extracted_data.get(category, "")).strip()
                
                # Normalizar fechas si el campo es una fecha
                if "fecha" in category or "periodo" in category:
                    original_value = normalize_date(original_value)
                    extracted_value = normalize_date(extracted_value)
                
                if original_value == extracted_value:
                    total_correct[category] += 1

# Calcular y mostrar los porcentajes de acierto
# Calculamos y mostramos la tasa de acierto para cada categoría de entidad.
print("Tasa de acierto por categoría:")
for category in categories:
    if total_entries[category] > 0:
        accuracy = (total_correct[category] / total_entries[category]) * 100
        print(f"{category}: {accuracy:.2f}%")
    else:
        print(f"{category}: No hay datos")

# Calcular y mostrar la tasa de acierto global
if global_total > 0:
    global_accuracy = (global_correct / global_total) * 100
    print(f"Tasa de acierto global: {global_accuracy:.2f}%")
else:
    print("No hay datos para calcular la tasa de acierto global")

# Calcular y mostrar los porcentajes de documentos con diferentes niveles de acierto
thresholds = [0.20, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00]
threshold_counts = {threshold: 0 for threshold in thresholds}

for accuracy in document_accuracies:
    for threshold in thresholds:
        if accuracy >= threshold:
            threshold_counts[threshold] += 1

print("Porcentaje de documentos con diferentes niveles de acierto:")
for threshold in thresholds:
    percentage = (threshold_counts[threshold] / len(document_accuracies)) * 100
    print(f"≥ {int(threshold * 100)}%: {percentage:.2f}%")