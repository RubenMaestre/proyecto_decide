# Proyecto de Extracción de Información de Facturas Eléctricas

## Introducción al Desafío

El reto consiste en ser capaces de extraer algunos campos determinados de información de facturas eléctricas. Estas facturas se presentan en distintos formatos, todas ellas en archivos PDF, pero con un orden y disposición diferente de los campos. Sin embargo, la información de las facturas, aun con variaciones, es esencialmente la misma, y algunos de estos campos comunes deben ser obtenidos de ellas.

El objetivo es ser capaces de crear un método que extraiga la información de las facturas de la forma más genérica posible, no personalizando para ningún tipo de plantilla de factura. De hecho, cuando se evalúe vuestro código, habrá tipos diferentes de plantillas de facturas que no estaban presentes en el dataset de entrenamiento, por lo que hacer algo genérico es fundamental. De cada una de las facturas habrá que obtener los mismos campos, de forma que se convierta esta información desestructurada presente en la factura en una información estructurada, con el mismo formato para todas ellas.

Se pueden desarrollar los métodos que se consideren oportunos para conseguir este objetivo, con los modelos que creáis más convenientes. La única limitación será utilizar Python.

## Dataset de Entrenamiento

Nos proporcionan el siguiente material, que no puedo subir a GitHub debido a que los datos no son míos, sino del desafío:

- **Facturas PDF**: El dataset de entrenamiento consiste en un número de facturas en formato PDF.
- **Archivos JSON**: Correspondientes a las facturas, contienen la información que se quiere extraer.

El dataset de entrenamiento está formado por 1000 facturas, numeradas desde `factura_0.pdf` hasta `factura_999.pdf`. Cada una de ellas tiene un archivo JSON correspondiente con el mismo nombre, cambiando sólo la extensión, con los 19 campos que hay que extraer de las facturas.

## Campos a Extraer

Los campos a extraer de las facturas son los siguientes:

- **Nombre del cliente** (`nombre_cliente`)
- **DNI del cliente** (`dni_cliente`)
- **Calle del cliente** (`calle_cliente`)
- **Código postal del cliente** (`cp_cliente`)
- **Población del cliente** (`población_cliente`)
- **Provincia del cliente** (`provincia_cliente`)
- **Nombre de la empresa comercializadora** (`nombre_comercializadora`)
- **CIF de la comercializadora** (`cif_comercializadora`)
- **Dirección de la comercializadora** (`dirección_comercializadora`)
- **Código postal de la comercializadora** (`cp_comercializadora`)
- **Población de la comercializadora** (`población_comercializadora`)
- **Provincia de la comercializadora** (`provincia_comercializadora`)
- **Número de factura** (`número_factura`)
- **Inicio del periodo de facturación** (`inicio_periodo`)
- **Fin del periodo de facturación** (`fin_periodo`)
- **Importe de la factura** (`importe_factura`)
- **Fecha del cargo** (`fecha_cargo`)
- **Consumo en el periodo** (`consumo_periodo`)
- **Potencia contratada** (`potencia_contratada`)

## Formatos de Datos

- **Fechas**: Formato `DD.MM.YYYY`, por ejemplo, `07.02.2016`.
- **Valores numéricos con decimales**: Separados por una coma, por ejemplo, `191,32`.

Para cualquier duda sobre los formatos, se pueden consultar los archivos JSON del dataset de entrenamiento para comprobar cuál es el formato utilizado.

## Consideraciones Adicionales

- Los archivos PDF contienen texto dentro de ellos. Existen librerías en Python que permiten extraer el texto directamente del archivo PDF. Sin embargo, hay que tener cuidado ya que no todas las librerías extraen el mismo texto ni en el mismo formato.
- No se considerará la diferencia entre letras mayúsculas y minúsculas, pero sí se considerarán caracteres diferentes si llevan tilde o si no. Usad la codificación apropiada para tenerlo en cuenta.
- Es posible que en algunas facturas haya datos que faltan, mostrados como `XX`. No obstante, no debería ocurrir en ninguno de los casos de los datos que tenéis que obtener. El resto de información de las facturas es irrelevante.

## Descarga del Dataset de Entrenamiento

El dataset de entrenamiento puede descargarse en el siguiente enlace: [training.zip](#)

El dataset de test está formado por otras 1000 facturas, también numeradas desde `factura_0.pdf` hasta `factura_999.pdf`. Los archivos JSON que generéis deberían seguir la misma nomenclatura, también usada en el dataset de entrenamiento, de forma que los archivos tengan el mismo nombre pero diferente extensión.

## Estructura del Proyecto

Muchos de los datos comienzan en una carpeta que estará nombrada en varios archivos llamada `TRAINING` que, por temas de privacidad y derechos, no puedo subir a GitHub.

```plaintext
|-- proyecto_decide/
    |-- training/
        |-- factura_0.pdf
        |-- factura_0.json
        |-- ...
        |-- factura_999.pdf
        |-- factura_999.json
```

Entonces ahora quiero explicar que es mi primera experiencia en NLP.
Había visto alguna cosa en mi formación en Hack a Boss, pero nada como lo que he ido descubriendo enfrentándome a este desafío.

También quiero decir, por si alguien quiere parar de leer ya, que no he conseguido hacerlo. Con el mejor modelo trabajado he obtenido alrededor de un 40% de acierto a la hora de acertar todos los datos... es cierto que en campos como DNI o CIF he obtenido casi un 100% de acierto, pero en otros, como el de los municipios prácticamente un 0%.

Entonces, con todo esto, lo primero que me vino a la cabeza fue utilizar REGEX, pero prácticamente lo deseché porque entendía que iba a ser una locura hacerlo así si el formato de las facturas podía cambiar.

Por lo tanto, la idea era entrenar un modelo que fuera capaz de reconocer texto de una factura, y a partir de ahí extraer los datos.

Lo he intentado con 3 modelos diferentes. Comencé con spaCy, y he ido acabando con BERT y ROBERTA.

Curiosamente con spaCy he obtenido mejores métricas que con BERT y ROBERTA, que deberían haber funcionado mejor. Quizás no he sido capaz de dar con la tecla con BERT. Tenía claro como hacerlo, pero a la hora de ponerlo a funcionar nunca conseguí acertar cómo hacerlo. Creo que el modelo entrenaba muy bien, pero la tokenización del mismo, separando por sílabas por así decirlo, desglosando los números en unidades más pequeñas, a la hora de validar el modelo y encontrar patrones me generaba muchos problemas. También me enfrenté a problemas de desbalanceamiento de categorías... por un lado me salía la categoría que no quería, la del texto en general casi con un 85% y las que yo quería que entrenara no llegaban al 15%. Tuve que enfrentarme a esos problemas.

## El proyecto

Esta sería digamos la carpeta que os vais a encontrar, menos los archivos de training. Cada una de las carpetas es un digamos un proyecto dentro del proyecto.
Primero comencé con cuadernos, luego seguí con libretas y finalicé con bert.
La que mejor resultados, o digamos funciona por así decirlo, es lo que hay en la carpeta libretas.

```plaintext
|-- proyecto_decide/
    |-- training/
        |-- factura_0.pdf
        |-- factura_0.json
        |-- ...
        |-- factura_999.pdf
        |-- factura_999.json
    |-- cuadernos/
    |-- libretas/
    |-- bert/
    |-- decide/
```

### Descripción de las Carpetas

- **training**: Contiene los datos del desafío (facturas en formato PDF y sus correspondientes archivos JSON). Por temas de protección de los datos no puedo compartir esta carpeta.
- **decide**: Es el entorno virtual que he creado para trabajar en el proyecto. No la he subido porque no tiene sentido, pero sí está el archivo `requirements.txt`.
- **cuadernos**: Aquí desarrollo el modelo utilizando spaCy. En esta carpeta abordo los primeros desafíos y voy experimentando hasta comprender mejor el proceso.
- **libretas**: Después de trabajar en `cuadernos`, aquí aplico lo aprendido y continúo desarrollando el modelo spaCy hasta obtener buenas métricas en algunos datos, aunque con un acierto global del 40%. Es en esta fase cuando decido cambiar de modelo.
- **bert**: Contiene el trabajo realizado con BERT y posteriormente con ROBERTA. Aquí intento encontrar un método que reconozca el texto de la factura y extraiga los datos sin depender de patrones específicos.

### Experiencia con el Proyecto

La idea era entrenar un modelo que fuera capaz de reconocer texto de una factura, y a partir de ahí extraer los datos. Lo he intentado con tres modelos diferentes: comencé con spaCy y luego trabajé con BERT y ROBERTA.

En `cuadernos` y `libretas` desarrollo el modelo spaCy. En `cuadernos` abordo los primeros desafíos y experimento hasta comprender mejor el proceso. En `libretas`, con más conocimiento, trabajo hasta obtener mejores métricas en algunos datos, aunque con un acierto global del 40%. Decidí entonces cambiar de modelo y busqué uno que reconociera el texto y extrajera los datos, en lugar de crear patrones como en spaCy. Sin embargo, al trabajar con BERT y luego con ROBERTA, encontré problemas de tokenización y desbalanceo de categorías, que no pude resolver completamente en el tiempo disponible.

### Modelo spaCy

Comencé normalizando las plantillas de texto mediante el script normalizacion_plantillas.py, que elimina saltos de línea y reduce espacios dobles a simples, asegurando un formato uniforme para el procesamiento. Luego, con el script preprocesamiento.py, generé documentos de texto a partir de plantillas, sustituyendo los marcadores de posición por datos específicos cargados desde archivos JSON, creando así 20,000 documentos necesarios para entrenar el modelo.

Para el entrenamiento del modelo, utilicé el script entrenamiento.py, donde procesé los documentos y sus entidades para crear archivos binarios compatibles con spaCy. Estos archivos fueron utilizados para entrenar y validar el modelo, que fue inicializado con un modelo preentrenado en español y ajustado para reconocer las etiquetas de entidades que me interesaban. Evalué el modelo continuamente para ajustar los parámetros y mejorar su precisión.

Una vez entrenado el modelo, utilicé validacion.py para evaluar su rendimiento, comparando las entidades predichas con las reales en los datos de validación. Para probar el modelo con texto de entrada manual, desarrollé analizar_texto.py, permitiendo introducir texto por terminal y observar las entidades reconocidas por el modelo.

Finalmente, con el script extraccion_pdf.py, extraje texto de archivos PDF de facturas, lo limpié y normalicé para su procesamiento. Utilicé el modelo entrenado para detectar patrones específicos en el texto y guardé los resultados en archivos JSON mediante el script app.py. Para medir la precisión del modelo, medicion.py comparó los resultados extraídos por el modelo con los datos originales, calculando la tasa de acierto para cada categoría de entidad y una tasa de acierto global.

### Modelo BERT y ROBERTA

En la carpeta bert está el trabajo realizado con estos modelos. Inicialmente comencé con BERT, preparando todo gracias al trabajo previo con spaCy. Comencé a entrenar el modelo, pero me llevó mucho tiempo y no conseguí los resultados esperados.

Me di cuenta que BERT tokenizaba las palabras en unidades más pequeñas, lo que dificultaba el reconocimiento de ciertos patrones, como los DNI. Intenté con ROBERTA, que debería reconocer mejor esos patrones, pero también enfrenté problemas de tokenización y desbalanceo de categorías.

Intenté crear un tokenizador personalizado que no separara las palabras ni los números, pero tampoco funcionó. Por último, intenté balancear las categorías introduciendo más datos específicos, pero aún así no conseguí que el modelo reconociera correctamente los patrones deseados.

Así que en la carpeta BERT está el modelo trabajado. En un principio comencé a trabajar con BERT. Preparé todo, que gracias al trabajo de spaCy de tener ya posicionada y etiquetadas las categorías que iba a trabajar, lo que facilitaba mucho el trabajo. Me llevó un tiempo comprender la tokenización de BERT y comencé a entrenar. Esto sin duda ha sido lo peor. Me ha llevado muchísimas horas, he estado entrenando dos veces más de 60 horas el modelo. Y lo peor es darte cuenta cuando termina que no funciona todo lo bien que te gustaría que funcionara. Es muy frustrante. Imagino que habrán otras formas de hacerlo, tendré que investigar, o ver opciones de máquinas más potentes para trabajar, pero la verdad, es que esta parte ha sido sin duda la peor, la más desesperante. Os traslado esta valoración personal porque ha sido demoledora.

¿Qué me pasaba con BERT? Al tokenizar, separa todas las palabras en unidades más pequeñas. Y parece que el entrenamiento iba bien, se podían ajustar ciertas cosas, pero bueno, más o menos entiendo que el modelo lo hacía bien. ¿Qué es lo que pasaba? Si yo luego quería buscar un DNI que por ejemplo es 99888777Z, el modelo lo que hacía era desglosarlo en #99, #88, #87, #77, #Z... entonces no había forma de que me reconociera un DNI. Era capaz de reconocer texto, pero no era capaz de reconocer unidades. Al menos yo no he sabido hacerlo.

Estuve investigando y vi que con ROBERTA, que es otro modelo de BERT, se reconocen mejor ese tipo de datos. Entonces hice lo mismo que con BERT, aquí es verdad que lo que hice, porque me iba quedando sin tiempo, fue reducir el tiempo de entrenamiento con menos épocas, utilizando más batchs, y también reduciendo el número de archivos para el entrenamiento. Me pasó igual, no acababa de conseguir que me encontrara las categorías que yo quería al tokenizar. Realmente ha sido frustrante no encontrar la clave. Por un lado, pensaba que lo tenía todo bien hecho pero era incapaz de trasladar al modelo una vez entrenado lo que yo quería que hiciera, pero por otro lado, también era posible que no estuviese haciéndolo bien. A todo esto el tiempo se iba agotando...

Así que planteando la opción de que mi modelo estuviese bien, pero el problema fuera la tokenización, encontré la opción de hacer un tokenizador personalizado. Esto quiere decir, no separar las palabras ni los números. Es decir, si el nombre es Carlos, pues es Carlos, no #Car, #los, por ejemplo, y lo mismo con el DNI por poner el ejemplo. La idea era que entrenara con los datos sin tokenizarlos, a ver si así era más fácil encontrarlos. Tampoco funcionó...

Entonces me di cuenta que igual era posible que no lo reconociera porque tenía más etiquetas O que datos de las categorías que quería buscar, estaba desbalanceado. Fui introduciendo mejoras en el entrenamiento para que fuera descartando partes de las etiquetas O, y ya por último lo que hice fue meterle más datos de todas las demás categorías para intentar balancearlo. Pero con el único objetivo de ver si el modelo era capaz de identificar un DNI, si una vez que yo le introducía el texto y lo leía, identificaba un DNI pues estaría más cerca. Pongo DNI porque es un patrón mucho más reconocible que por ejemplo una dirección o un nombre. Pero nada, no funcionó y el tiempo se ha agotado...

Tengo varias ideas para abordarlo. Desde la premisa de que lo primero que tiene que hacer, que en principio es lo que hace, es "aprender" a leer. Estuve viendo y tiene opciones como CHATBOTS, que no solo sea reconocer texto, sino que le puedes hacer preguntas. Entonces estuve viendo de plantearlo desde esa premisa, te paso un texto y te pregunto ¿Hay un DNI en el texto? Si lo hubiese, podemos llevar esa respuesta a un JSON con los datos. Igual no es una forma ortodoxa de solucionar el problema, pero igual podría haber resuelto el problema... no lo sé, porque ya el entrenamiento así eran más de 60 horas de nuevo, tenía que tener mucho texto y los datos que quería reconocer.

### Conclusión

A pesar de no haber logrado el objetivo final, esta experiencia ha sido muy enriquecedora. He aprendido mucho sobre NLP y la complejidad de trabajar con diferentes modelos y técnicas. Espero que esta base pueda servir a otros para encontrar nuevas ideas y abordar el problema de una manera más efectiva.

Me gusta contar todo, intentar meter todos los detalles y aunque no pueda trasladar los sentimientos de frustración y rabia al ver las horas de entrenamiento que se iban, ver cómo luego no funcionaba, volver a investigar a ver qué se hacía mal, toda esa impotencia de tener el código, de tener los datos y, bueno, en tu cabeza es sencillo leer un texto y sacar un DNI por ejemplo, pero ser incapaz de hacerlo con un modelo que en principio debe hacerlo pues genera mucha rabia... pero también es verdad que te ayuda a enfrentarte al desafío, a buscar soluciones, a intentar resolver el problema y encontrar la forma de solucionarlo. Aunque no ha sido mi caso, como digo el tiempo se me ha terminado, tengo que reconocer que ha sido una gran experiencia.

