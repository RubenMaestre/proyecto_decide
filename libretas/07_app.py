# app.py
# Este será el archivo principal que configuraremos para realizar toda la operación de extracción de datos de las facturas.
# Utilizaremos este script para cargar el modelo entrenado de spaCy, aplicar el modelo a las facturas originales extraídas,
# detectar patrones específicos en el texto, y guardar los resultados en archivos JSON.

import spacy
import os
import json
import re
from dateutil.parser import parse, ParserError

# Ruta del modelo entrenado
model_path = 'modelo_entrenado'

# Cargar el modelo entrenado
nlp = spacy.load(model_path)

# Directorios
input_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/facturas/'
output_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/validaciones/'

# Asegurarse de que el directorio de salida existe
os.makedirs(output_dir, exist_ok=True)

# Categorías de entidades
# Estas son las etiquetas que el modelo debe reconocer y extraer del texto de las facturas
categories = [
    "nombre_cliente", "dni_cliente", "calle_cliente", "cp_cliente", "población_cliente",
    "provincia_cliente", "nombre_comercializadora", "cif_comercializadora", "dirección_comercializadora",
    "cp_comercializadora", "población_comercializadora", "provincia_comercializadora", "número_factura",
    "inicio_periodo", "fin_periodo", "importe_factura", "fecha_cargo", "consumo_periodo", "potencia_contratada"
]

# Para facilitar la ayuda al modelo, podemos de alguna forma "seleccionar" que datos extraer de los textos
# que sean más reconocibles mediante expresiones regulares u otras. 
# Lista de provincias españolas con sus variaciones
provincias_espanolas = [
    "Álava", "Araba", "Albacete", "Alicante", "Alacant", "Almería", "Asturias", "Ávila", "Badajoz", 
    "Baleares", "Illes Balears", "Barcelona", "Burgos", "Cáceres", "Cádiz", "Cantabria", "Castellón", 
    "Castelló", "Ciudad Real", "Córdoba", "Cuenca", "Gerona", "Girona", "Granada", "Guadalajara", 
    "Guipúzcoa", "Gipuzkoa", "Huelva", "Huesca", "Jaén", "La Coruña", "A Coruña", "La Rioja", "Las Palmas",
    "León", "Lleida", "Lugo", "Madrid", "Málaga", "Murcia", "Navarra", "Nafarroa", "Orense", "Ourense", 
    "Palencia", "Pontevedra", "Salamanca", "Santa Cruz de Tenerife", "Segovia", "Sevilla", "Soria", 
    "Tarragona", "Teruel", "Toledo", "Valencia", "Valladolid", "Vizcaya", "Bizkaia", "Zamora", "Zaragoza"
]

# Funciones para detectar patrones en el texto
def detect_dni(text):
    match = re.search(r'\b\d{8}[A-Z]\b', text, re.IGNORECASE)
    return match.group() if match else ""

def detect_cp(text):
    match = re.search(r'\b\d{5}\b', text)
    return match.group() if match else ""

def detect_nombre_cliente(text):
    # Eliminar ocurrencias de NIF y DNI antes de buscar el nombre
    text = re.sub(r'\bNIF\s*\d{8}[A-Z]\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDNI\s*\d{8}[A-Z]\b', '', text, flags=re.IGNORECASE)
    # Expresión regular para nombres españoles con soporte para mayúsculas, minúsculas y acentos, y sin números ni caracteres no válidos
    pattern = r'\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [A-ZÁÉÍÓÚÑ][a-záéíóúñ]+\b|\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+ [a-záéíóúñ]+ [A-ZÁÉÍÓÚÑáéíóúñ]+\b'
    matches = re.findall(pattern, text)
    for match in matches:
        if not re.search(r'\d', match):
            # Verificar que no haya letras sueltas separadas por espacios, puntos o comas
            if not re.search(r'([A-Za-zÁÉÍÓÚÑáéíóúñ]\s{1}[A-Za-zÁÉÍÓÚÑáéíóúñ])|([A-Za-zÁÉÍÓÚÑáéíóúñ]\.{1}[A-Za-zÁÉÍÓÚÑáéíóúñ])|([A-Za-zÁÉÍÓÚÑáéíóúñ],{1}[A-Za-zÁÉÍÓÚÑáéíóúñ])', match):
                return match
    return ""

# Aquí para ayudar a detectar las provincias
def detect_provincia(text):
    for provincia in provincias_espanolas:
        if re.search(r'\b' + re.escape(provincia.lower()) + r'\b', text.lower()):
            return provincia
    return ""

# Aquí introducimos el patrón que siguen más o menos las facturas
def detect_numero_factura(text):
    match = re.search(r'\b[A-Z0-9]{10,13}\b', text, re.IGNORECASE)
    return match.group() if match else ""

# Aquí para la potencia contratada
def detect_potencia(text):
    match = re.search(r'\b\d{1,2},\d{3}\b', text)
    return match.group() if match else ""

# Esto es para el importe de las facturas...
def detect_importe(text):
    match = re.search(r'\b\d{1,3},\d{2}\b', text)
    return match.group() if match else ""

# Aquí para detectar fechas
def detect_fecha(text):
    try:
        parsed_date = parse(text, dayfirst=True)
        return parsed_date.strftime("%d.%m.%Y")
    except (ParserError, ValueError):
        return ""

# Este es para la detección del consumo, ya que hay muchos números enteros, le ayudamos con el kWh
def detect_consumo(text):
    match = re.search(r'\b(\d{1,4}) kWh\b', text, re.IGNORECASE)
    return match.group(1) if match else ""

# Aquí para detectar direcciones
def detect_direccion(text):
    pattern = (
        r'\b(?:C\/|Calle|Avenida|Avda\.|Av\.|Plaza|Paseo|Pje\.|Pl\.|Parque|Camino|Carretera|Cami|Urb\.)\s+[\w\s]+(?:,\s*\d+|\s+\d+)?\s*(?:[-ºªA-Za-z0-9\s,]*)?'
    )
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group() if match else ""

# Función para limpiar el nombre de cliente, algunas veces se cuela la palabra NIF
def clean_nombre_cliente(nombre):
    # Eliminar cualquier ocurrencia de NIF o DNI y lo que venga después
    nombre = re.sub(r'\bNIF\s*\d{8}[A-Z]\b', '', nombre, flags=re.IGNORECASE)
    nombre = re.sub(r'\bDNI\s*\d{8}[A-Z]\b', '', nombre, flags=re.IGNORECASE)
    # Eliminar caracteres no deseados
    nombre = re.sub(r'[^\w\sÁÉÍÓÚÑáéíóúñ]', '', nombre)
    # Eliminar espacios extra
    nombre = re.sub(r'\s+', ' ', nombre).strip()
    return nombre

# Función para validar y ajustar entidades según patrones conocidos
def validate_and_adjust_entities(entities, text):
    # Detectar patrones en el texto
    detected_entities = {
        "dni_cliente": detect_dni(text),
        "cp_cliente": detect_cp(text),
        "nombre_cliente": detect_nombre_cliente(text),
        "provincia_cliente": detect_provincia(text),
        "provincia_comercializadora": detect_provincia(text),
        "número_factura": detect_numero_factura(text),
        "potencia_contratada": detect_potencia(text),
        "importe_factura": detect_importe(text),
        "inicio_periodo": detect_fecha(text),
        "fin_periodo": detect_fecha(text),
        "fecha_cargo": detect_fecha(text),
        "consumo_periodo": detect_consumo(text),
        "dirección_comercializadora": detect_direccion(text)
    }

    # Ajustar entidades extraídas con las detectadas por patrones
    for key in detected_entities:
        if not entities[key] and detected_entities[key]:
            entities[key] = detected_entities[key]

    # Validar y ajustar el CP de la comercializadora... para que no coja el del cliente
    if detected_entities["dirección_comercializadora"]:
        context_text = text[text.find(detected_entities["dirección_comercializadora"]):]
        cp_comercializadora = detect_cp(context_text)
        if cp_comercializadora:
            entities["cp_comercializadora"] = cp_comercializadora

    # Limpiar el nombre del cliente
    entities["nombre_cliente"] = clean_nombre_cliente(entities["nombre_cliente"])
    
    return entities

# Función para procesar el texto y extraer las entities...
# Esta función utiliza el modelo spaCy para extraer entidades específicas del texto de las facturas
def extract_entities(text):
    doc = nlp(text)
    segments = text.split(' | ')
    entities = {category: "" for category in categories}
    
    for segment in segments:
        segment_doc = nlp(segment)
        for ent in segment_doc.ents:
            if ent.label_ in categories and not entities[ent.label_]:
                entities[ent.label_] = ent.text

    # Validar y ajustar las entities extraídas
    entities = validate_and_adjust_entities(entities, text)
    
    return entities

# Procesamos cada archivo en el directorio de entrada...
# Iteramos sobre todos los archivos en el directorio de entrada, leemos el texto,
# aplicamos el modelo para extraer entidades y guardamos los resultados en archivos JSON.
for filename in os.listdir(input_dir):
    if filename.endswith('.txt') or filename.endswith('.json'):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            if filename.endswith('.json'):
                data = json.load(file)
                text = data.get("text", "")
            else:
                text = file.read()

        # Extraer entidades del texto...
        extracted_entities = extract_entities(text)

        # Guardar los resultados en un archivo JSON
        output_filename = os.path.splitext(filename)[0] + '_result.json'
        output_path = os.path.join(output_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(extracted_entities, output_file, ensure_ascii=False, indent=4)

        print(f"Resultados guardados en {output_path}")
