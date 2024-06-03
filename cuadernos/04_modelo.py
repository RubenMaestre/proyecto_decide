# 04_modelo.py
# En este script vamos a cargar el modelo entrenado y procesar archivos JSON para extraer entidades nombradas.

import spacy
import json
import os

# Cargar el modelo entrenado
output_dir = "modelo_entrenado"
nlp = spacy.load(output_dir)

# Ruta de los archivos de entrada y salida
input_folder = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/training/json"
output_folder = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/cuadernos/results"
os.makedirs(output_folder, exist_ok=True)

# Procesar los primeros 5 archivos JSON
for i, json_file in enumerate(os.listdir(input_folder)):
    if i >= 5:
        break

    with open(os.path.join(input_folder, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Concatenar todo el texto del JSON para procesarlo
    text = " ".join(str(value) for value in data.values())
    
    # Procesar el texto con el modelo entrenado
    doc = nlp(text)
    
    # Extraer las entidades encontradas
    entidades = {ent.label_: ent.text for ent in doc.ents}
    
    # Guardar las entidades en un archivo JSON en la carpeta de resultados
    output_path = os.path.join(output_folder, json_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entidades, f, indent=4, ensure_ascii=False)

    print(f"Entidades del archivo {json_file} guardadas en {output_path}")

print("Procesamiento completado.")
