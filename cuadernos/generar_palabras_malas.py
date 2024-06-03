import json
import random
from nltk.corpus import words as nltk_words
import nltk

# Descargar el paquete de palabras en inglés si no está ya descargado
nltk.download('words')

# Lista inicial de palabras comunes en español
palabras_comunes = [
    "agua", "árbol", "casa", "ciudad", "familia", "gente", "hombre", "mujer", "niño", "niña", "comida", "bebida", "trabajo", "juego",
    "amor", "odio", "felicidad", "tristeza", "alegría", "dolor", "amistad", "enemigo", "sol", "luna", "estrella", "cielo", "tierra",
    "mar", "río", "montaña", "bosque", "flor", "animal", "perro", "gato", "pájaro", "pez", "ratón", "elefante", "tigre", "león",
    "caballo", "vaca", "oveja", "cerdo", "mono", "ciervo", "jirafa", "zebra", "cocodrilo", "serpiente", "sapo", "rana", "lagarto",
    "tortuga", "pato", "pollo", "gallina", "pavo", "ganso", "cisne", "águila", "halcón", "búho", "murciélago", "abeja", "mariposa",
    "mosca", "mosquito", "hormiga", "cucaracha", "araña", "escorpión", "cangrejo", "langosta", "camaron", "calamar", "pulpo", "ballena",
    "delfín", "tiburón", "foca", "pingüino", "leopardo", "pantera", "gato montés", "lince", "carro", "camión", "moto", "bicicleta",
    "avión", "barco", "tren", "metro", "tranvía", "autobús", "taxi", "señal de tráfico", "parque", "jardín", "torre", "rascacielos",
    "universidad", "hospital", "clínica", "farmacia", "supermercado", "tienda", "mercado", "restaurante", "café", "bar", "cine",
    "teatro", "museo", "biblioteca", "iglesia", "templo", "mezquita", "sinagoga", "cementerio", "castillo", "palacio", "fortaleza",
    "muralla", "puerta", "ventana", "techo", "pared", "suelo", "escalera", "ascensor", "lavabo", "baño", "cocina", "sala", "comedor",
    "dormitorio", "habitación", "balcón", "terraza", "garaje", "sótano", "ático", "desván", "ventilador", "calefacción",
    "aire acondicionado", "radiador", "chimenea", "fuego", "luz", "oscuridad", "sombra", "color", "rojo", "azul", "verde", "amarillo",
    "naranja", "rosa", "morado", "marrón", "negro", "blanco", "gris", "dorado", "plateado", "cobre", "bronce", "oro", "plata",
    "diamante", "rubí", "esmeralda", "zafiro", "topacio", "amatista", "turquesa", "jade", "cristal", "vidrio", "plástico", "metal",
    "madera", "piedra", "cemento", "ladrillo", "mármol", "granito", "arena", "tierra", "barro", "arcilla", "carbón", "petróleo",
    "gas", "plomo", "hierro", "acero", "aluminio", "níquel", "cromo", "zinc", "estaño", "magnesio", "mercurio", "plutonio",
    "uranio", "radón", "cobalto", "vanadio", "tungsteno", "litio", "potasio", "sodio", "calcio", "estroncio", "bario", "radio",
    "escandio", "itrio", "lantano", "cerio", "praseodimio", "neodimio", "prometio", "samario", "europio", "gadolinio", "terbio",
    "disprosio", "holmio", "erbio", "tulio", "iterbio", "lutecio", "hafnio", "tantalio", "renio", "osmium", "iridio", "platino",
    "oro", "plata", "cobre", "níquel", "plomo", "estaño", "zinc", "cadmio", "mercurio", "franco", "radón", "actinio", "torio",
    "protactinio", "uranio", "neptunio", "plutonio", "americio", "curio", "berkelio", "californio", "einsteinio", "fermio",
    "mendelevio", "nobelio", "lawrencio", "rutherfordio", "dubnio", "seaborgio", "bohrio", "hassio", "meitnerio", "darmstadtio",
    "roentgenio", "copernicio", "nihonio", "flerovio", "moscovio", "livermorio", "tenesino", "oganesson",
    "pero", "este", "aquello", "aquella", "aquellas", "doscientos", "mil", "millón", "tres", "sus", "las", "además", "electricidad", "voltaje", "resistencia", "circuito", "electrón", "átomo", "ciencia", "tecnología", "energía", "fuerza",
    "campo", "magnético", "física", "química", "biología", "laboratorio", "experimento", "teoría", "práctica", "innovación",
    "programación", "software", "hardware", "computadora", "red", "internet", "algoritmo", "base de datos", "inteligencia artificial",
    "robot", "automatización", "ingeniería", "matemáticas", "fórmula", "ecuación", "gravedad", "relatividad", "célula", "genética",
    "microbio", "nanotecnología", "biotecnología", "batería", "cargador", "eléctrico", "electromagnetismo", "fotón", "neutrón",
    "protones", "nanómetro", "milivoltio", "sensor", "dispositivo", "transistor", "microprocesador", "fibra óptica", "semiconductor",
    "plasma", "fusión", "fisión", "radiación", "frecuencia", "amperio", "ómhio", "vatio", "kilovatio", "megavatio", "gigavatio",
    "teravatio", "conductor", "aislante", "resistor", "capacitancia", "inductancia", "transformador", "osciloscopio", "generador",
    "motor", "fotovoltaico", "solar", "eólico", "térmico", "hidráulico", "nuclear", "geotérmico", "biomasa", "hidrógeno", "fuel",
    "combustible", "energético", "sostenibilidad", "medioambiente", "contaminación", "emisiones", "carbono", "oxígeno", "hidrógeno",
    "nitrógeno", "helio", "luz", "ultravioleta", "infrarrojo", "rayos", "gamma", "rayos X", "electromagnético", "capa", "ozono",
    "efecto", "invernadero", "clima", "meteorología", "atmosférico", "presión", "viento", "tormenta", "huracán", "tornado", "ciclón",
    "maremoto", "tsunami", "terremoto", "sismo", "volcán", "lava", "magma", "erupción", "tectónica", "placa", "crustal", "litosfera",
    "manto", "núcleo", "geología", "mineral", "roca", "sedimento", "fósil", "paleontología", "arqueología", "antropología", "sociología",
    "psicología", "neurociencia", "medicina", "anatomía", "fisiología", "patología", "inmunología", "virología", "bacteriología",
    "micología", "parasitología", "epidemiología", "farmacología", "toxología", "nutrición", "dietética", "genética", "bioquímica",
    "biología molecular", "biología celular", "biología estructural", "ecología", "zoología", "botánica", "microbiología", "astrobiología",
    "evolución", "ecosistema", "biodiversidad", "clonación", "transgénico", "modificación genética", "terapia génica", "bioética",
    "biomedicina", "bioinformática", "biotecnología industrial", "biotecnología ambiental", "biotecnología roja", "biotecnología blanca",
    "biotecnología verde", "bioingeniería", "biofísica", "biomateriales", "biomecánica", "biorremediación", "biosensores", "bioseparación",
    "bioeconomía", "bioenergía", "biohidrógeno", "biofotónica", "biomecatrónica", "biomímesis", "biomímesis", "bionanotecnología",
    "biomimética", "biotecnología marina", "biotecnología agrícola", "biotecnología forestal", "biotecnología animal", "biotecnología humana",
    "biotecnología industrial", "biotecnología enzimática", "biotecnología de alimentos", "biotecnología de bebidas", "biotecnología de medicamentos",
    "biotecnología de biocombustibles", "biotecnología de bioplásticos", "biotecnología de bioproductos", "biotecnología de biofertilizantes",
    "biotecnología de bioinsecticidas", "biotecnología de biopesticidas", "biotecnología de bioremediación", "biotecnología de biodesulfuración",
    "biotecnología de biolixiviación", "biotecnología de biosíntesis", "biotecnología de biocatálisis", "biotecnología de bioconversión",
    "biotecnología de biodesulfurización", "biotecnología de biodesulfurización", "biotecnología de biodesulfurización"
]

# Obtener palabras del corpus nltk
palabras_ingles = nltk_words.words()
random.shuffle(palabras_ingles)

# Seleccionar al azar entre 2000 y 3500 palabras
num_palabras = random.randint(2000, 3500)
palabras_irrelevantes = random.sample(palabras_comunes, min(1800, len(palabras_comunes))) + random.sample(palabras_ingles, min(1200, len(palabras_ingles)))
palabras_irrelevantes = random.sample(palabras_irrelevantes, num_palabras)
random.shuffle(palabras_irrelevantes)

# Guardar en archivo JSON
output_file = 'C:/Users/34670/Desktop/python/Hack a boss/proyecto_decide/cuadernos/json_categoria/datos_texto_no_validos.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(palabras_irrelevantes, f, ensure_ascii=False, indent=4)

print(f"Archivo '{output_file}' generado con éxito.")


