# normalizacion_plantillas.py
# Este script se encarga de normalizar el texto en archivos de plantilla.
# No vamos a poner el texto en minúsculas ni a eliminar acentos porque queremos que el modelo de SPACY entrene con ellos también. 
# Específicamente, eliminamos saltos de línea y reducimos espacios dobles a simples, con el objetivo de uniformar
# el formato de los textos dentro de un directorio específico.

import os

def normalizar_texto(texto):
    # Eliminamos los saltos de línea para que el texto sea una sola línea continua...
    texto = texto.replace('\n', ' ')
    # Este bucle se asegura de que no queden espacios dobles, convirtiéndolos en espacios simples...
    while '  ' in texto:
        texto = texto.replace('  ', ' ')
    return texto

# Definimos el directorio donde se encuentran las plantillas que vamos a procesar.
plantillas_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/plantillas/"

# Iteramos sobre cada archivo en el directorio de plantillas...
for filename in os.listdir(plantillas_dir):
    if filename.endswith(".txt"):  # Nos aseguramos de procesar solo archivos de texto.
        path = os.path.join(plantillas_dir, filename)  # Obtenemos la ruta completa del archivo.
        with open(path, "r") as file:
            texto_plantilla = file.read()  # Leemos el contenido del archivo.
        texto_normalizado = normalizar_texto(texto_plantilla)  # Normalizamos el texto usando nuestra función.
        # Ahora escribimos el texto normalizado de vuelta en el mismo archivo.
        with open(path, "w") as file:
            file.write(texto_normalizado)
        print(f"Plantilla '{filename}' procesada y normalizada.")  # Confirmamos que el archivo ha sido procesado.

