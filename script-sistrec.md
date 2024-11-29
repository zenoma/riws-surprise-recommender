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


## Conclusión
