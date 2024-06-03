import spacy
from spacy.tokens import DocBin

# Cargar los datos binarios
def load_data(data_path, nlp):
    doc_bin = DocBin().from_disk(data_path)
    return list(doc_bin.get_docs(nlp.vocab))

# Inicializar spaCy con un modelo preentrenado (cualquiera que sea compatible con tu idioma)
nlp = spacy.blank('es')  # Puedes usar 'es_core_news_md' o cualquier otro modelo

# Cargar los datos de entrenamiento y validación
train_data_path = 'train_data.spacy'
val_data_path = 'val_data.spacy'

train_docs = load_data(train_data_path, nlp)
val_docs = load_data(val_data_path, nlp)

# Función para visualizar el contenido de los documentos
def visualize_docs(docs):
    for doc in docs:
        print(f"Texto: {doc.text}")
        print("Entidades:")
        for ent in doc.ents:
            print(f" - {ent.text} ({ent.label_})")
        print("\n")

# Visualizar algunos ejemplos de los datos de entrenamiento y validación
print("Datos de Entrenamiento:")
visualize_docs(train_docs[:5])  # Mostrar los primeros 5 documentos

print("\nDatos de Validación:")
visualize_docs(val_docs[:5])  # Mostrar los primeros 5 documentos
