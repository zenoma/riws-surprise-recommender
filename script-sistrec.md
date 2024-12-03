# PRÁCTICA DE SISTEMAS DE RECOMENDACIÓN
**Universidade da Coruña**  
**Adrián García Vázquez**  
**Correo:** adrian.gvazquez@udc.es  

---

## Introducción
En esta memoria se documenta el desarrollo de la práctica del bloque de sistemas de recomendación en la asignatura de RIWS. La práctica se centra en la implementación de modelos de recomendación utilizando el paquete `scikit-surprise`, abarcando desde la carga y exploración de los datos hasta la aplicación de varios algoritmos y la evaluación de sus resultados.

---

## Paso 1: Descarga y extracción del dataset 
### Descripción
En este paso, se descarga el dataset de la URL proporcionada, se guarda localmente y se descomprime para obtener el archivo `ratings.csv`. Este archivo contiene las valoraciones de los usuarios sobre diversas películas.


### Análisis del Resultado
Después de ejecutar esta función, el archivo `ratings.csv` se encuentra en la carpeta `ml-latest-small`. Esto permite que se pueda cargar en un `DataFrame` para su análisis posterior.

---

## Paso 2: Carga de los datos en un DataFrame
### Descripción
Cargamos el archivo `ratings.csv` en un `DataFrame` utilizando la librería pandas. Esto facilita el manejo y la exploración de los datos.


### Análisis del Resultado
Los datos se encuentran dentro del rango mencionado y esperado. No existen números duplicados ni valores vacíos(NA).

## Paso 3: Eliminación de películas y usuarios con pocas puntuaciones
### Descripción
Eliminamos los películas (movieId) con menos de 10 puntuaciones, ya que es importante garantizar que cada película tenga un volumen significativo de interacciones para análisis confiables. Posteriormente, aplicamos el mismo criterio a los usuarios (userId) eliminando aquellos con menos de 10 puntuaciones. Esto ayuda a mantener únicamente a los usuarios más activos en el análisis.

### Análisis del Resultado

Películas restantes: Calculamos el número de películas que superan el umbral.
Usuarios restantes: Obtenemos cuántos usuarios permanecen tras aplicar ambos filtros.
Puntuaciones restantes: El tamaño final del dataset tras estos pasos.

## Paso 4: Histograma del número de puntuaciones por usuario y película
#### Descripción

Generamos dos histogramas para analizar la distribución de las puntuaciones:

    Número de puntuaciones por usuario: Mostramos cuántos usuarios realizaron un cierto número de puntuaciones.
    Número de puntuaciones por película: Mostramos cuántos películas recibieron un cierto número de puntuaciones.

### Análisis del Resultado

En el histograma del número de puntuaciones por usuario, podemos observar que los usuarios tienden a dar menos de 200 valoraciones.
Para el histograma del número de puntuaciones por película, observamos que los películas en general suelen tener pocas puntuaciones.

## Paso 5: Histograma de la media de puntuaciones por usuario y película
### Descripción

Creamos dos histogramas para analizar la distribución de las medias de puntuación:

    Media de puntuaciones por usuario: Calculamos la media de las puntuaciones dadas por cada usuario y representamos su distribución.
    Media de puntuaciones por película: Calculamos la media de las puntuaciones recibidas por cada película y representamos su distribución.


####  Análisis del Resultado

La tendencia de los usuarios es a valorar las películas entre valores de 3.0 y 4.5. Esto no nos indica mucho sobre el comportamiento de los usuarios, pues puede que esté sesgado la calidad del catálogo al que los usuarios acceden.
Por otra parte, la tendencia de la puntuación por película también varía mayormente entre 2.5 y 4.0. Las películas no suelen llevar puntuaciones extremas, es muy difícil obtener un 1 o un 5.

## Paso 6: Diagrama de barras de la distribución de las puntuaciones
### Descripción

Representamos en un diagrama de barras la cantidad de veces que se ha dado cada puntuación.

### Análisis del Resultado

Podemos observar que las puntuaciones más usadas varían entre 3 y 4. Esto nos indica que los usuarios tienden a valorar productos de forma "neutra", dado que los valores entre 3 y 4 son valores correctos al nivel de satisfacción tanto de usuario como de ítem.

## Conclusión
