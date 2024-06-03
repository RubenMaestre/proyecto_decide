# preprocesamiento.py
# Este script se encarga de generar documentos de texto a partir de plantillas,
# sustituyendo los marcadores de posición (placeholders) por datos específicos
# cargados desde archivos JSON. Vamos a generar 20,000 documentos para tener 
# suficientes datos de facturas para entrenar nuestro modelo de SPACY.

import os
import json
import random

# Directorios de entrada y salida...
# Cargamos las plantillas que hemos generado y donde tenemos las etiquetas llamadas placeholders.
# Estos placeholders serán sustituidos por los datos de los archivos JSON.
# Así, podemos generar una cantidad infinita de plantillas con los datos disponibles.
plantillas_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/plantillas/"
json_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/json_categoria/"
output_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/datos/"

# Diccionario de placeholders...
# Este diccionario mapea los placeholders a sus correspondientes archivos JSON.
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

# Función para cargar datos de JSON...
# Aquí cargamos los datos desde los archivos JSON en un diccionario.
def cargar_datos_json(json_dir, json_files):
    datos = {}
    for placeholder, key in json_files.items():
        file_name = f"{key}.json"
        with open(os.path.join(json_dir, file_name), 'r', encoding='utf-8') as f:
            datos[placeholder] = json.load(f)
    return datos

# Función para generar un texto con datos aleatorios y capturar las entidades...
# Sustituimos los placeholders en la plantilla por datos aleatorios del JSON correspondiente
# y capturamos la posición de las entidades para usarlas en el entrenamiento del modelo...
def generar_texto_con_datos(plantilla, datos_json):
    entities = []
    texto_final = ""
    offset = 0

    for placeholder, key in json_files.items():
        while placeholder in plantilla:
            valor = random.choice(datos_json[placeholder])
            start = plantilla.find(placeholder)
            end = start + len(valor)
            
            texto_final += plantilla[:start] + valor
            plantilla = plantilla[start + len(placeholder):]
            
            # Ajuste de entidades y desplazamiento de posiciones
            for entity in entities:
                if entity[0] >= offset + start:
                    entity[0] += len(valor) - len(placeholder)
                if entity[1] >= offset + start:
                    entity[1] += len(valor) - len(placeholder)
                    
            entities.append([offset + start, offset + start + len(valor), key])
            offset += start + len(valor)
    
    texto_final += plantilla

    return texto_final, entities

# Función principal para generar documentos...
# Esta función coordina la carga de datos, selección de plantillas, generación de textos y escritura en archivos.
def generar_documentos(plantillas_dir, json_dir, output_dir, json_files, num_docs=20000):
    # Cargar datos JSON...
    datos_json = cargar_datos_json(json_dir, json_files)
    
    # Obtener plantillas...
    plantillas = [f for f in os.listdir(plantillas_dir) if f.endswith('.txt')]
    
    # Generar documentos...
    documentos_descartados = 0
    for i in range(num_docs):
        plantilla_file = random.choice(plantillas)
        with open(os.path.join(plantillas_dir, plantilla_file), 'r', encoding='utf-8') as f:
            plantilla = f.read()
        
        texto, entities = generar_texto_con_datos(plantilla, datos_json)
        
        if not entities:
            documentos_descartados += 1
            continue
        
        data_entry = {
            "text": texto,
            "entities": entities
        }
        
        output_file = os.path.join(output_dir, f"documento_{i+1}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_entry, f, ensure_ascii=False, indent=4)

    print(f"Total de documentos descartados: {documentos_descartados}")

# Ejecutar la función principal...
# Creamos el directorio de salida si no existe y llamamos a la función para generar los documentos.
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

generar_documentos(plantillas_dir, json_dir, output_dir, json_files, num_docs=20000)
