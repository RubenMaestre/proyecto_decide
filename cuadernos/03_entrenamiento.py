# 03_entrenamiento.py
# En este script vamos a entrenar un modelo NER usando spaCy con los datos preprocesados y generar un reporte de clasificación.

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from sklearn.metrics import classification_report
from spacy.util import compounding, minibatch

# Cargar los datos de entrenamiento y validación
train_data_path = 'train_data.spacy'
val_data_path = 'val_data.spacy'

# Cargar datos binarios
def load_data(data_path):
    doc_bin = DocBin().from_disk(data_path)
    return list(doc_bin.get_docs(nlp.vocab))

# Inicializar spaCy con un modelo preentrenado más robusto
nlp = spacy.load('es_core_news_lg')

# Cargar los datos
train_data = load_data(train_data_path)
val_data = load_data(val_data_path)

# Añadir el componente NER al pipeline si no existe
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

# Añadir las etiquetas al componente NER
for doc in train_data:
    for ent in doc.ents:
        ner.add_label(ent.label_)

# Configurar los parámetros de entrenamiento con una tasa de aprendizaje ajustada
optimizer = nlp.create_optimizer()
optimizer.learn_rate = 0.001

# Función para evaluar el modelo
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
    
    # Asegurarse de que ambas listas tienen la misma longitud
    min_length = min(len(y_true), len(y_pred))
    y_true = y_true[:min_length]
    y_pred = y_pred[:min_length]
    
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return report

# Entrenar el modelo
n_iter = 50
best_f1_score = 0.0
patience = 5
no_improvement_counter = 0

for itn in range(n_iter):
    losses = {}
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        examples = []
        for doc in batch:
            example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
            examples.append(example)
        nlp.update(examples, sgd=optimizer, drop=0.5, losses=losses)  # Ajusta el dropout aquí
    
    print(f"Iteración {itn + 1}, Pérdidas: {losses}")

    # Evaluar el modelo en los datos de validación cada 5 iteraciones
    if (itn + 1) % 5 == 0:
        report = evaluate_model(nlp, val_data)
        print(f"Reporte de clasificación para la iteración {itn + 1}:\n{report}")
        
        # Medir F1-score y comparar con el mejor
        current_f1_score = float(report.split()[-2])
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            no_improvement_counter = 0
            # Guardar el mejor modelo
            nlp.to_disk('best_model')
        else:
            no_improvement_counter += 1
        
        if no_improvement_counter >= patience:
            print("Early stopping debido a la falta de mejora en el rendimiento.")
            break

# Guardar el modelo entrenado final
output_dir = 'modelo_entrenado'
nlp.to_disk(output_dir)
print(f"Modelo guardado en '{output_dir}'")

# Evaluar el modelo final
nlp_trained = spacy.load(output_dir)
final_report = evaluate_model(nlp_trained, val_data)
print(f"Reporte de clasificación final:\n{final_report}")
