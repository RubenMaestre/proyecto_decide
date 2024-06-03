# 04_entrenamiento.py
# Comentar que este se ha hecho con ROBERTA y la configuración actual es con ROBERTA y el tokenizador personalizado.
# Pero que se ha hecho también con BERT y ROBERTA sin el tokenizador personalizado. Se pueden modificar sin problemas en el script.

import os
import json
import torch
from collections import Counter
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, RobertaConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch.nn as nn

# Definir la clase del dataset
class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach()
        return item

    def __len__(self):
        return len(self.labels)

# Mapeo de etiquetas a índices
label2id = {
    "O": 0,
    "B-nombre_cliente": 1, "I-nombre_cliente": 2,
    "B-dni_cliente": 3, "I-dni_cliente": 4,
    "B-calle_cliente": 5, "I-calle_cliente": 6,
    "B-cp_cliente": 7, "I-cp_cliente": 8,
    "B-poblacion_cliente": 9, "I-poblacion_cliente": 10,
    "B-provincia_cliente": 11, "I-provincia_cliente": 12,
    "B-nombre_comercializadora": 13, "I-nombre_comercializadora": 14,
    "B-cif_comercializadora": 15, "I-cif_comercializadora": 16,
    "B-direccion_comercializadora": 17, "I-direccion_comercializadora": 18,
    "B-cp_comercializadora": 19, "I-cp_comercializadora": 20,
    "B-poblacion_comercializadora": 21, "I-poblacion_comercializadora": 22,
    "B-provincia_comercializadora": 23, "I-provincia_comercializadora": 24,
    "B-numero_factura": 25, "I-numero_factura": 26,
    "B-inicio_periodo": 27, "I-inicio_periodo": 28,
    "B-fin_periodo": 29, "I-fin_periodo": 30,
    "B-importe_factura": 31, "I-importe_factura": 32,
    "B-fecha_cargo": 33, "I-fecha_cargo": 34,
    "B-consumo_periodo": 35, "I-consumo_periodo": 36,
    "B-potencia_contratada": 37, "I-potencia_contratada": 38,
}

# Aquí validamos y limpiamos los datos
def validate_and_clean_data(encodings):
    for key, val in encodings.items():
        for i in range(len(val)):
            if isinstance(val[i], str):
                try:
                    val[i] = json.loads(val[i])
                except json.JSONDecodeError:
                    raise ValueError(f"Error decoding JSON in {key} at index {i}: {val[i]}")
            if not all(isinstance(x, int) for x in val[i]):
                raise ValueError(f"Non-integer value found in {key} at index {i}: {val[i]}")
    return encodings

# Función para cargar los datos preprocesados
def load_preprocessed_data(data_dir, tokenizer, max_len=512):
    encodings = {'input_ids': [], 'attention_mask': [], 'token_type_ids': []}
    labels = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith('.json'):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    input_ids = tokenizer.convert_tokens_to_ids(item['tokens'])
                    # Asegurarse de que los tokens estén dentro del rango del vocabulario
                    input_ids = [i if i < tokenizer.vocab_size else tokenizer.unk_token_id for i in input_ids]
                    
                    attention_mask = item['attention_mask']
                    token_type_ids = item.get('token_type_ids', [0]*len(input_ids))
                    item_labels = [label2id[label] for label in item['labels']]

                    # Truncar y padear a la longitud máxima
                    input_ids = input_ids[:max_len] + [0]*(max_len - len(input_ids))
                    attention_mask = attention_mask[:max_len] + [0]*(max_len - len(attention_mask))
                    token_type_ids = token_type_ids[:max_len] + [0]*(max_len - len(token_type_ids))
                    item_labels = item_labels[:max_len] + [-100]*(max_len - len(item_labels))

                    encodings['input_ids'].append(input_ids)
                    encodings['attention_mask'].append(attention_mask)
                    encodings['token_type_ids'].append(token_type_ids)
                    labels.append(item_labels)

    # Validar y limpiar los datos
    encodings = validate_and_clean_data(encodings)

    # Convertir las listas a tensores de PyTorch
    try:
        encodings = {key: torch.tensor(val) for key, val in encodings.items()}
        labels = torch.tensor(labels)
    except Exception as e:
        print(f"Error converting to tensors: {e}")
        print(f"encodings: {encodings}")
        print(f"labels: {labels}")
        raise

    return encodings, labels

# Calcular los pesos inversos para las etiquetas y ajustar la etiqueta "O"
def calculate_weights(labels, o_weight_ratio=0.15):
    labels_flat = [label for sublist in labels for label in sublist]
    label_counts = Counter(labels_flat)
    total_counts = sum(label_counts.values())
    label_weights = {label: total_counts / count for label, count in label_counts.items()}
    
    # Ajustar el peso de la etiqueta "O"
    total_weight = sum(label_weights.values())
    adjusted_o_weight = total_weight * o_weight_ratio / (1 - o_weight_ratio)
    label_weights[0] = adjusted_o_weight  # Cambiado de "O" a 0 para coincidir con la clave en label2id

    # Asegurarse de que todas las etiquetas están presentes en los pesos
    weights = []
    for label in sorted(label2id.values()):
        if label in label_weights:
            weights.append(label_weights[label])
        else:
            weights.append(1.0)
    
    return torch.tensor(weights, dtype=torch.float)

# Función para computar las métricas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Remover los tokens de padding
    true_labels = [label for pred, label in zip(preds.flatten(), labels.flatten()) if label != -100]
    true_preds = [pred for pred, label in zip(preds.flatten(), labels.flatten()) if label != -100]

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='macro')
    acc = accuracy_score(true_labels, true_preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Configuraciones
data_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/datos_roberta/'
model_name = 'PlanTL-GOB-ES/roberta-base-bne'
output_dir = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/archivos_roberta_2/'
tokenizer_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/tokenizer/"
num_labels = len(label2id)

# Desactivar advertencia de symlinks
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Configuración personalizada para ajustar el dropout
config = RobertaConfig.from_pretrained(
    model_name,
    hidden_dropout_prob=0.1,  # Dropout en las capas ocultas
    attention_probs_dropout_prob=0.1,  # Dropout en la atención
    num_labels=num_labels
)

# Cargar el modelo y el tokenizer de RoBERTa con la configuración personalizada
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_dir, use_fast=True)
model = RobertaForTokenClassification.from_pretrained(model_name, config=config)

# Cargar los datos preprocesados
encodings, labels = load_preprocessed_data(data_dir, tokenizer)

# Calcular los pesos
weights = calculate_weights(labels, o_weight_ratio=0.15)
loss_fct = nn.CrossEntropyLoss(weight=weights)

# Verificar que las longitudes coincidan
assert len(encodings['input_ids']) == len(labels), "Inconsistencia en el número de muestras entre encodings y labels"

# Dividir en conjuntos de entrenamiento y evaluación
indices = list(range(len(labels)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_encodings = {key: val[train_indices] for key, val in encodings.items()}
val_encodings = {key: val[val_indices] for key, val in encodings.items()}
train_labels = labels[train_indices]
val_labels = labels[val_indices]

train_dataset = NERDataset(train_encodings, train_labels)
val_dataset = NERDataset(val_encodings, val_labels)

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_steps=74,
    eval_steps=74,
    eval_strategy="steps",
    load_best_model_at_end=True,
    save_total_limit=2,
    save_strategy="steps"
)

# Trainer con la función de pérdida personalizada
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

# Entrenar el modelo
trainer.train()

# Guardar el modelo y el tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
