# 05_validazcion.py
# En este script vamos a extraer texto de archivos PDF, limpiar y normalizar el texto, tokenizarlo y predecir etiquetas usando un modelo BERT entrenado.

import os
import re
import json
import fitz  # PyMuPDF
import spacy
from transformers import BertTokenizerFast, BertForTokenClassification
import torch
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
    text = re.sub(r'[^\w\s.,€]', '', text)  # Eliminar caracteres no deseados, mantener números, letras, puntos y comas
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
        r'\b\d{1,2}\s+de\s+\w+\s+de\s+\d{4}\b',              # Formato: 2 de octubre de 1991
        re.IGNORECASE)

    return date_pattern.sub(replace_date, text)

# Función para tokenizar el texto usando spaCy
def tokenize_text(text, nlp):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

# Función para predecir etiquetas usando el modelo entrenado
def predict_labels(tokens, model, tokenizer):
    results = {}
    for token in tokens:
        encoded_input = tokenizer([token], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**encoded_input)
        predictions = torch.argmax(outputs.logits, dim=-1)
        label = model.config.id2label[predictions[0][0].item()]
        if label != "O":
            field_name = label[2:]
            if field_name not in results:
                results[field_name] = []
            results[field_name].append(token)
    return results

# Cargar el modelo y el tokenizador entrenado
model = BertForTokenClassification.from_pretrained("./model")
tokenizer = BertTokenizerFast.from_pretrained("./model")
nlp = spacy.load("es_core_news_sm")

# Directorios
pdf_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/training'
output_folder = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/cuadernos/results'
os.makedirs(output_folder, exist_ok=True)

# Procesar y predecir para cada PDF en el directorio
for i in range(5):  # Limitar a los primeros 5 archivos
    pdf_path = os.path.join(pdf_folder, f'factura_{i}.pdf')
    json_output_path = os.path.join(output_folder, f'factura_{i}.json')

    if not os.path.exists(pdf_path):
        print(f"Archivo no encontrado: {pdf_path}")
        continue

    # Extraer texto y limpiar
    pdf_text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(pdf_text)
    normalized_text = normalize_dates(cleaned_text)
    tokens = tokenize_text(normalized_text, nlp)

    # Predecir etiquetas y guardar resultados
    results = predict_labels(tokens, model, tokenizer)
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"Datos predichos guardados en {json_output_path}")

print("Procesamiento completado.")
