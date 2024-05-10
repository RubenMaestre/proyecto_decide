import pdfplumber
import os
import json

def extract_text_from_pdf(pdf_path):
    text_content = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text_content.append(page.extract_text())
    return "\n".join(text_content)

# Ruta a la carpeta donde están los PDFs
pdf_folder = 'training/pdf'
extracted_texts = []

# Itera sobre cada archivo en la carpeta de PDFs
for pdf_file in os.listdir(pdf_folder):
    if pdf_file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, pdf_file)
        extracted_text = extract_text_from_pdf(pdf_path)
        extracted_texts.append({'file_name': pdf_file, 'text': extracted_text})

# Ruta para guardar los textos extraídos en formato JSON
output_folder = 'training/json'
os.makedirs(output_folder, exist_ok=True)  # Asegura que la carpeta existe

# Guardar los textos extraídos en un archivo JSON dentro de la carpeta de JSONs
output_file = os.path.join(output_folder, 'extracted_texts.json')
with open(output_file, 'w') as outfile:
    json.dump(extracted_texts, outfile, indent=4)
