# 01_preprocesamiento.py
# En este script vamos a extraer, limpiar y normalizar el texto de archivos PDF para su posterior uso.

import fitz  # PyMuPDF
import os
import re
from dateutil.parser import parse

# Función para extraer texto de un PDF usando PyMuPDF (fitz)
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Función para limpiar el texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Eliminar URLs
    text = re.sub(r'\.{2,}', '.', text)  # Eliminar secuencias repetidas de puntos
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar secuencias repetidas de espacios
    text = re.sub(r'\bx,xx\b', '', text)  # Eliminar valores placeholder x,xx
    text = re.sub(r'(?<!\d)€', '', text)  # Eliminar € que no tienen un número antes
    text = re.sub(r'\bpágina \d+\b', '', text)  # Eliminar números de página
    text = re.sub(r'(?<!\d),\d{1,2}\b', '', text)  # Eliminar comas que no tienen un número antes y tienen solo 1 o 2 dígitos después
    text = re.sub(r'(?<=\s)[\.\,](?=\s)', '', text)  # Eliminar puntos y comas solitarios
    text = re.sub(r'[^\w\s.,€]', '', text)  # Eliminar caracteres no deseados, mantener números, letras, puntos, comas y €
    return text

# Función para normalizar las fechas a un formato estándar
def normalize_dates(text):
    def replace_date(match):
        date_str = match.group()
        try:
            date_obj = parse(date_str, dayfirst=True)
            return date_obj.strftime("%d.%m.%Y")
        except (ValueError, OverflowError):
            return date_str

    # Expresión regular para encontrar fechas en varios formatos
    date_pattern = re.compile(
        r'\b\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4}\b|'  # Formatos: 15/09/2024, 15-01-21, 15.01.21, 15 01 2021
        r'\b\d{4}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{1,2}\b|'      # Formato: 2024/09/17, 2024-09-17, 2024.09.17
        r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b|'              # Formato: 2 de octubre de 1991
        r'\b\d{8}\b',                                        # Formato: 27092018
        re.IGNORECASE)

    return date_pattern.sub(replace_date, text)

# Ruta a la carpeta de los PDFs y la carpeta de salida para los textos extraídos
pdf_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/training'
output_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/cuadernos/pre'
os.makedirs(output_folder, exist_ok=True)

# Procesar todos los PDFs en la carpeta
for pdf_file in os.listdir(pdf_folder):
    if not pdf_file.endswith('.pdf'):
        continue

    pdf_path = os.path.join(pdf_folder, pdf_file)
    
    # 1. Extraer texto del PDF
    text = extract_text_from_pdf(pdf_path)
    
    # 2. Limpiar el texto
    text = clean_text(text)
    
    # 3. Normalizar las fechas en el texto
    text = normalize_dates(text)
    
    # Guardar el texto limpiado en un archivo
    output_file = os.path.join(output_folder, pdf_file.replace('.pdf', '.txt'))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Texto extraído y limpiado de '{pdf_file}' guardado en '{output_file}'")

print("Procesamiento completado.")
