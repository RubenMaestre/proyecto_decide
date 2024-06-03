# 07_reentreno.py
# Aquí lo que vamos a hacer, viendo que tenemos algo desbalanceadas nuestras categorías y donde tenemos más datos en las etiquetas "O",
# he pensado que si entreno de nuevo al modelo nutriéndolo con los datos de los JSON que tienen cada una de las categorías,
# puede ayudar al modelo a comprender qué tipo de datos estamos buscando.

import os
import json
import random

# Directorio de los archivos JSON
json_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/json_categoria/"
output_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/nuevos_datos/"
output_file = os.path.join(output_dir, "nuevos.json")

# Diccionario de etiquetas basado en los nombres de archivo
json_labels = {
    "nombre_cliente": "nombre_cliente",
    "dni_cliente": "dni_cliente",
    "calle_cliente": "calle_cliente",
    "cp_cliente": "cp_cliente",
    "poblacion_cliente": "poblacion_cliente",
    "provincia_cliente": "provincia_cliente",
    "nombre_comercializadora": "nombre_comercializadora",
    "cif_comercializadora": "cif_comercializadora",
    "direccion_comercializadora": "direccion_comercializadora",
    "cp_comercializadora": "cp_comercializadora",
    "poblacion_comercializadora": "poblacion_comercializadora",
    "provincia_comercializadora": "provincia_comercializadora",
    "numero_factura": "numero_factura",
    "inicio_periodo": "inicio_periodo",
    "fin_periodo": "fin_periodo",
    "importe_factura": "importe_factura",
    "fecha_cargo": "fecha_cargo",
    "consumo_periodo": "consumo_periodo",
    "potencia_contratada": "potencia_contratada",
}

# Crear el directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Función para cargar, seleccionar y etiquetar datos
def cargar_seleccionar_etiquetar_datos(json_dir, json_labels, porcentaje=0.1):
    datos_etiquetados = []
    for file_name in os.listdir(json_dir):
        base_name = os.path.splitext(file_name)[0]
        if base_name in json_labels:
            label = json_labels[base_name]
            file_path = os.path.join(json_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                datos = json.load(file)
                seleccionados = random.sample(datos, int(len(datos) * porcentaje))
                for dato in seleccionados:
                    tokens = [dato]  # Mantener el dato completo como un token
                    etiquetas = [f"B-{label}"]
                    datos_etiquetados.append({"tokens": tokens, "labels": etiquetas})
    return datos_etiquetados

# Cargar, seleccionar y etiquetar los datos
datos_etiquetados = cargar_seleccionar_etiquetar_datos(json_dir, json_labels)

# Guardar los datos etiquetados en un archivo JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(datos_etiquetados, f, ensure_ascii=False, indent=4)

print(f"Datos etiquetados guardados en {output_file}")
