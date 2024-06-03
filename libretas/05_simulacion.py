# analizar_texto.py
# Este script permite ingresar texto por terminal y devuelve las entidades reconocidas
# por el modelo de reconocimiento de entidades (NER) entrenado con spaCy.
# Esto lo hago para resolver una primera duda: ¿qué tan bien funciona el modelo si le
# devuelvo un texto que yo escribo introduciendo datos manualmente?

import spacy

# Cargar el modelo entrenado...
# Asegúrate de que el directorio 'modelo_entrenado' contenga tu modelo entrenado.
output_dir = 'modelo_entrenado'
nlp = spacy.load(output_dir)

# Lista de categorías de entidades que queremos encontrar...
# Estas son las etiquetas que esperamos que el modelo reconozca en el texto.
categorias_entidades = [
    "nombre_cliente", "dni_cliente", "calle_cliente", "cp_cliente",
    "poblacion_cliente", "provincia_cliente", "nombre_comercializadora",
    "cif_comercializadora", "direccion_comercializadora", "cp_comercializadora",
    "poblacion_comercializadora", "provincia_comercializadora", "numero_factura",
    "inicio_periodo", "fin_periodo", "importe_factura", "fecha_cargo",
    "consumo_periodo", "potencia_contratada"
]

# Función para analizar el texto ingresado por terminal...
# Procesa el texto con el modelo spaCy y extrae las entidades reconocidas.
def analizar_texto(texto):
    doc = nlp(texto)
    entidades_encontradas = []
    for ent in doc.ents:
        if ent.label_ in categorias_entidades:
            entidades_encontradas.append((ent.text, ent.label_))
    return entidades_encontradas

# Ingresar texto por terminal...
# Permite al usuario introducir un texto y analiza las entidades presentes en él.
if __name__ == "__main__":
    texto_usuario = input("Introduce el texto a analizar: ")
    entidades = analizar_texto(texto_usuario)
    if entidades:
        print("Entidades reconocidas:")
        for entidad, etiqueta in entidades:
            print(f"{etiqueta}: {entidad}")
    else:
        print("No se encontraron entidades reconocidas.")
