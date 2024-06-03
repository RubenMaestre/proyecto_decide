# 02_tokenizador.py
# Aquí lo que hemos hecho es intentar personalizar nuestro tokenizador para que no separase en unidades más pequeñas las palabras.
# Si no quieres utilizar este script y prefieres usar el tokenizador normal de BERT, no lo ejecutes.

import os
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# Ruta al archivo de corpus
corpus_file = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/datos/corpus.txt"
tokenizer_output_dir = "C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/bert/tokenizer/"

# Crear un tokenizer basado en WordLevel
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))

# Utilizamos un pre-tokenizador basado en espacios para separar las palabras
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Configuramos el decodificador
tokenizer.decoder = decoders.WordPiece()

# Configuramos el procesador de plantilla para incluir tokens especiales
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)

# Configuramos el entrenador del tokenizador
trainer = trainers.WordLevelTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# Entrenamos el tokenizador con el archivo de corpus
tokenizer.train([corpus_file], trainer)

# Guardamos el tokenizador en formato JSON
if not os.path.exists(tokenizer_output_dir):
    os.makedirs(tokenizer_output_dir)
tokenizer.save(os.path.join(tokenizer_output_dir, "tokenizer.json"))

# Guardamos el vocabulario en formato vocab.txt
vocab = tokenizer.get_vocab()
vocab_file_path = os.path.join(tokenizer_output_dir, "vocab.txt")
with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
    for token, index in sorted(vocab.items(), key=lambda item: item[1]):
        vocab_file.write(f"{token}\n")

print("Tokenizador entrenado y guardado correctamente.")
