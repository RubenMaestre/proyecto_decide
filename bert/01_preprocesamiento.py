# 01_preprocesamiento.py
# En este script preparamos los archivos necesarios para entrenar modelos BERT y ROBERTA. 
# Utilizamos plantillas y datos JSON para generar textos con entidades etiquetadas.

import os
import json
import random

# Directorios de entrada y salida
plantillas_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/plantillas/"
json_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/json_categoria/"
output_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/datos/"
corpus_file = os.path.join(output_dir, "corpus.txt")  # Archivo para el corpus
entidades_file = os.path.join(output_dir, "entidades.json")  # Archivo para datos etiquetados

# Diccionario de placeholders que relaciona cada placeholder con su correspondiente archivo JSON
json_files = {
    "placeholder_01": "nombre_cliente",
    "placeholder_02": "dni_cliente",
    "placeholder_03": "calle_cliente",
    "placeholder_04": "cp_cliente",
    "placeholder_05": "poblacion_cliente",
    "placeholder_06": "provincia_cliente",
    "placeholder_07": "nombre_comercializadora",
    "placeholder_08": "cif_comercializadora",
    "placeholder_09": "direccion_comercializadora",
    "placeholder_10": "cp_comercializadora",
    "placeholder_11": "poblacion_comercializadora",
    "placeholder_12": "provincia_comercializadora",
    "placeholder_13": "numero_factura",
    "placeholder_14": "inicio_periodo",
    "placeholder_15": "fin_periodo",
    "placeholder_16": "importe_factura",
    "placeholder_17": "fecha_cargo",
    "placeholder_18": "consumo_periodo",
    "placeholder_19": "potencia_contratada",
}

# Aquí cargamos los datos de los archivos JSON en un diccionario
def cargar_datos_json(json_dir, json_files):
    datos = {}
    for placeholder, key in json_files.items():
        file_name = f"{key}.json"
        file_path = os.path.join(json_dir, file_name)
        with open(file_path, 'r', encoding='utf-8') as f:
            datos[placeholder] = json.load(f)
            print(f"Cargado {file_name} con {len(datos[placeholder])} entradas.")
    return datos

# Esta función normaliza el texto en archivos de plantilla
# Reemplaza saltos de línea y dobles espacios por una barra vertical y un solo espacio
def normalizar_texto(texto):
    return texto.replace('\n', '|').replace('  ', ' ')

# Aquí ajustamos las posiciones de las entidades que están fuera de rango en el texto generado
def ajustar_entidades_fuera_de_rango(texto, entities):
    max_len = len(texto)
    adjusted_entities = []
    for start, end, label in entities:
        if start >= max_len:
            start = max_len - 1
        if end > max_len:
            end = max_len
        adjusted_entities.append([start, end, label])
    return adjusted_entities

# En esta función generamos un texto con datos aleatorios y capturamos las entidades correspondientes
def generar_texto_con_datos(plantilla, datos_json):
    entities = []
    texto_final = plantilla
    offset = 0

    for placeholder, key in json_files.items():
        while placeholder in texto_final:
            valor = random.choice(datos_json[placeholder])
            start = texto_final.find(placeholder)
            end = start + len(placeholder)

            # Actualizamos texto_final reemplazando el placeholder por el valor real
            texto_final = texto_final[:start] + valor + texto_final[end:]

            # Registramos la entidad con su posición en el texto
            entities.append([start, start + len(valor), key])

            # Actualizamos el offset para la siguiente búsqueda
            offset = start + len(valor)

    # Verificamos los placeholders restantes en el texto
    for placeholder in json_files.keys():
        if placeholder in texto_final:
            print(f"Advertencia: {placeholder} no ha sido reemplazado en el texto final.")

    return texto_final, entities

# Función principal donde generamos los documentos y creamos el corpus y el archivo de entidades
def generar_documentos_y_corpus(plantillas_dir, json_dir, output_dir, corpus_file, entidades_file, json_files, num_docs=10000):
    # Primero cargamos los datos JSON
    datos_json = cargar_datos_json(json_dir, json_files)
    
    # Aquí obtenemos la lista de archivos de plantilla
    plantillas = [f for f in os.listdir(plantillas_dir) if f.endswith('.txt')]
    
    # Ahora generamos los documentos y creamos el corpus y el archivo de entidades
    documentos_descartados = 0
    with open(corpus_file, 'w', encoding='utf-8') as corpus, open(entidades_file, 'w', encoding='utf-8') as entidades:
        all_entities = []
        for i in range(num_docs):
            plantilla_file = random.choice(plantillas)
            with open(os.path.join(plantillas_dir, plantilla_file), 'r', encoding='utf-8') as f:
                plantilla = f.read()
            
            plantilla_normalizada = normalizar_texto(plantilla)
            texto, entities = generar_texto_con_datos(plantilla_normalizada, datos_json)
            
            if not entities:
                documentos_descartados += 1
                continue
            
            # Ajustamos las entidades fuera de rango
            entities = ajustar_entidades_fuera_de_rango(texto, entities)
            
            data_entry = {
                "text": texto,
                "entities": entities
            }
            
            # Guardamos el documento JSON (opcional, puedes comentar estas líneas si no necesitas guardar los JSONs)
            output_file = os.path.join(output_dir, f"documento_{i+1}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data_entry, f, ensure_ascii=False, indent=4)

            # Escribimos el texto en el corpus
            corpus.write(texto + '\n')

            # Agregamos las entidades al archivo de entidades
            all_entities.append(data_entry)
        
        # Guardamos todas las entidades en un archivo JSON
        json.dump(all_entities, entidades, ensure_ascii=False, indent=4)

    print(f"Total de documentos descartados: {documentos_descartados}")

# Aquí verificamos si el directorio de salida existe, y si no, lo creamos
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Finalmente, ejecutamos la función principal
generar_documentos_y_corpus(plantillas_dir, json_dir, output_dir, corpus_file, entidades_file, json_files, num_docs=10000)
