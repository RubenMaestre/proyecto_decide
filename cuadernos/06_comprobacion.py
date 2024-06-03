# 06_comprobacion.py
# En este script vamos a cargar el modelo y el tokenizador entrenado para comprobar manualmente la clasificación de datos introducidos por el usuario.

import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# Cargar el modelo y el tokenizador entrenado
model = BertForTokenClassification.from_pretrained("./model")
tokenizer = BertTokenizerFast.from_pretrained("./model")

# Mapa de etiquetas
id2label = {v: k for k, v in model.config.label2id.items()}

# Función para predecir etiquetas usando el modelo entrenado
def predict_label(text, model, tokenizer):
    encoded_input = tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encoded_input)
    predictions = torch.argmax(outputs.logits, dim=-1)
    label = model.config.id2label[predictions[0][0].item()]
    return label

def main():
    while True:
        user_input = input("Introduce un dato (o 'salir' para terminar): ").strip()
        if user_input.lower() == 'salir':
            break

        label = predict_label(user_input, model, tokenizer)

        if label == "O":
            print(f"El dato '{user_input}' no fue reconocido.")
        else:
            field_name = label[2:]
            print(f"El dato '{user_input}' fue reconocido como: {field_name}")

if __name__ == "__main__":
    main()
