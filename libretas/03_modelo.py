# modelo.py
# Este script utiliza spaCy para entrenar un modelo de reconocimiento de entidades (NER).
# El objetivo es extraer automáticamente campos específicos de facturas eléctricas en PDF.
# Dado que las facturas pueden variar en formato y disposición de los campos, necesitamos
# un método genérico que pueda manejar distintos tipos de plantillas de facturas y extraer
# la información necesaria de manera consistente.

import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from sklearn.metrics import classification_report
from spacy.util import compounding, minibatch
from spacy.lookups import load_lookups

# Cargar los datos de entrenamiento y validación
train_data_path = 'train_data.spacy'
val_data_path = 'val_data.spacy'

# Función para cargar datos binarios...
# Cargamos los datos de entrenamiento y validación desde archivos binarios de spaCy (DocBin).
def load_data(data_path, nlp):
    doc_bin = DocBin().from_disk(data_path)
    return list(doc_bin.get_docs(nlp.vocab))

# Inicializar spaCy con un modelo preentrenado en español...
# Usamos 'es_core_news_md' que es un modelo de tamaño medio para probar su eficacia. 
# Existe también el spacy.load('es_core_news_lg') para modelo grande
# y también el spacy.load('es_core_news_sm') para un modelo más pequeño.
nlp = spacy.load('es_core_news_md')

# Eliminar el componente 'matcher' si existe en el pipeline... me ha dado varias veces error 
# y eliminándolo se soluciona el problema y se ejecuta el modelo.
if 'matcher' in nlp.pipe_names:
    nlp.remove_pipe('matcher')

# Cargar las tablas de lemas necesarias para el lematizador...
# Esto mejora la precisión al normalizar palabras a su forma base.
lookups = load_lookups(lang="es", tables=["lemma_rules", "lemma_index", "lemma_exc", "lemma_rules_groups"])
nlp.get_pipe("lemmatizer").initialize(nlp.vocab, lookups=lookups)

# Cargar los datos de entrenamiento y validación...
train_data = load_data(train_data_path, nlp)
val_data = load_data(val_data_path, nlp)

# Añadir el componente NER al pipeline si no existe...
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner', last=True)
else:
    ner = nlp.get_pipe('ner')

# Añadir las etiquetas al componente NER...
# Aquí añadimos las etiquetas de las entidades que queremos que el modelo reconozca.
for doc in train_data:
    for ent in doc.ents:
        ner.add_label(ent.label_)

# Configurar los parámetros de entrenamiento...
# Ajustamos la tasa de aprendizaje para mejorar la convergencia del modelo.
optimizer = nlp.create_optimizer()
optimizer.learn_rate = 0.0005 # He probado con varias... con 0.001, con 0.0001... 

# Función para evaluar el modelo...
# Esta función evalúa la precisión del modelo comparando las entidades predichas con las reales.
# Esto es importante porque bueno, una de las veces estuve más de 4 horas esperando que el modelo terminara
# de entrenar y luego dio error en las métricas. 
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

# Entrenar el modelo...
# Aumentamos el número de iteraciones para asegurar una mejor convergencia. He probado con 10, 20... 100... 
n_iter = 50
# El evaluate_every en 1 lo he puesto para saber rápidamente si las evaluaciones las hacía bien y que no hubiese ningún error.
# Quizás es más interesante poner 10 psi vas a hacer 50 iteraciones para ir viendo la evaluación o 5 si haces 20 iteraciones...
evaluate_every = 1 
best_f1_score = 0.0
patience = 8 # esto es para que el modelo se detenga si no mejora... así no hay que esperar hasta 50 si ya en el 30 ve que no mejora
no_improvement_counter = 0

for itn in range(n_iter):
    losses = {}
    # Ajustamos el tamaño del batch para mejorar el rendimiento del entrenamiento.
    batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
    for batch in batches:
        examples = []
        for doc in batch:
            example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
            examples.append(example)
        # Ajustamos el dropout para prevenir sobreajuste. Igual, he probado con 0.3, 0.5... 
        nlp.update(examples, sgd=optimizer, drop=0.4, losses=losses)
    
    print(f"Iteración {itn + 1}, Pérdidas: {losses}")

    # Evaluar el modelo cada iteración...
    report = evaluate_model(nlp, val_data)
    print(f"Reporte de clasificación para la iteración {itn + 1}:\n{report}")
    
    # Medir F1-score y comparar con el mejor
    lines = report.split('\n')
    avg_line = [line for line in lines if 'avg' in line.lower()][0]
    current_f1_score = float(avg_line.split()[-2])
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
