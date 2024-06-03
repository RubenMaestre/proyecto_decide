# validacion.py
# Este script se utiliza para evaluar un modelo de reconocimiento de entidades (NER) entrenado con spaCy.
# Cargará el modelo que ya hemos entrenado y los datos de validación para evaluar el rendimiento del modelo.

import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report

# Función para cargar datos binarios...
# Cargamos los datos de validación desde un archivo binario de spaCy (DocBin).
def load_data(data_path, nlp):
    doc_bin = DocBin().from_disk(data_path)
    return list(doc_bin.get_docs(nlp.vocab))

# Inicializar spaCy y cargar el modelo entrenado...
# Asegúrate de que el directorio 'modelo_entrenado' contenga tu modelo entrenado.
output_dir = 'modelo_entrenado'
nlp = spacy.load(output_dir)

# Cargar los datos de validación
val_data_path = 'val_data.spacy'
val_data = load_data(val_data_path, nlp)

# Función para evaluar el modelo...
# Esta función evalúa la precisión del modelo comparando las entidades predichas con las reales.
def evaluate_model(nlp, data):
    y_true = []
    y_pred = []
    labels = list(nlp.get_pipe("ner").labels)
    for doc in data:
        gold_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        pred_doc = nlp(doc.text)
        pred_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in pred_doc.ents]
        
        y_true.extend([label for _, _, label in gold_entities])
        y_pred.extend([label for _, _, label in pred_entities])
    
    # Asegurar que y_true y y_pred tengan la misma longitud ¡Importante!
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]

    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return report

# Evaluar el modelo final
final_report = evaluate_model(nlp, val_data)
print(f"Reporte de clasificación final:\n{final_report}")
