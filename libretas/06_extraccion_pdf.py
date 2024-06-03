# extraccion_pdf.py
# Este script se utiliza para extraer texto de archivos PDF de facturas y limpiarlo.
# Una vez entrenado el modelo y confirmado que reconoce las entidades correctamente,
# procedemos con la extracción de los datos de las facturas originales.
# Vamos a utilizar la librería fitz (PyMuPDF) que parece dar buen rendimiento...

import fitz  # PyMuPDF
import os
import re
from dateutil.parser import parse, ParserError

# Función para extraer texto de un PDF usando PyMuPDF (fitz)
# Abrimos el PDF y extraemos el texto de cada página.
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Función para normalizar fechas
def normalize_dates(text):
    # Expresión regular para encontrar fechas en varios formatos
    date_patterns = [
        (r'\b(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2,4})\b', "%d/%m/%Y"),  # Formatos: 15/09/2024, 15-01-21, 15.01.21
        (r'\b(\d{2,4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})\b', "%d/%m/%Y"),  # Formato: 2024/09/17, 2024-09-17, 2024.09.17
        (r'\b(\d{8})\b', "%d/%m/%Y"),                                      # Formato: 27092018
        (r'\b(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\b', "%d/%m/%Y")         # Formato: 2 de octubre de 1991
    ]

    def replace_date(match, date_format):
        date_str = match.group()
        try:
            # Parse the date and format it as dd/mm/yyyy
            parsed_date = parse(date_str, dayfirst=True)
            return parsed_date.strftime(date_format)
        except (ParserError, ValueError):
            return date_str

    for pattern, date_format in date_patterns:
        text = re.sub(pattern, lambda match: replace_date(match, date_format), text, flags=re.IGNORECASE)

    return text

# Función para limpiar el texto
# Aplicamos varias reglas de limpieza para preparar el texto extraído.
def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Eliminar URLs
    text = re.sub(r'\.{2,}', '.', text)  # Eliminar secuencias repetidas de puntos
    text = re.sub(r'\s+', ' ', text).strip()  # Eliminar secuencias repetidas de espacios
    text = re.sub(r'\bx,xx\b', '', text)  # Eliminar valores placeholder x,xx
    text = re.sub(r'\bpágina \d+\b', '', text)  # Eliminar números de página
    text = re.sub(r'(?<=\s)[\.\,](?=\s)', '', text)  # Eliminar puntos y comas solitarios
    text = re.sub(r'[^\w\s.,€|-]', '', text)  # Eliminar caracteres no deseados, mantener números, letras, puntos, comas, € y guiones, y el símbolo |
    return text

# Ruta a la carpeta de los PDFs y la carpeta de salida para los textos extraídos
pdf_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/training'
output_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/libretas/facturas'
os.makedirs(output_folder, exist_ok=True)

# Procesar todos los PDFs en la carpeta
# Iteramos sobre todos los archivos PDF en la carpeta especificada.
for pdf_file in os.listdir(pdf_folder):
    if not pdf_file.endswith('.pdf'):
        continue

    pdf_path = os.path.join(pdf_folder, pdf_file)
    
    # 1. Extraer texto del PDF
    text = extract_text_from_pdf(pdf_path)
    
    # 2. Reemplazar saltos de línea con el símbolo |
    text = text.replace('\n', ' | ')
    
    # 3. Limpiar el texto
    text = clean_text(text)
    
    # 4. Normalizar las fechas en el texto
    text = normalize_dates(text)
    
    # Guardar el texto limpiado en un archivo
    output_file = os.path.join(output_folder, pdf_file.replace('.pdf', '.txt'))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"Texto extraído y limpiado de '{pdf_file}' guardado en '{output_file}'")

print("Procesamiento completado.")