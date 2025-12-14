# 📊 Estadística GUI en Python (Tkinter)

Aplicación de escritorio desarrollada en **Python + Tkinter** para el **análisis estadístico descriptivo e inferencial**, visualización de datos y modelado básico, a partir de datos tabulares (CSV, Excel o datos pegados desde Excel/Sheets).

El proyecto está orientado a **estudiantes, docentes y analistas de datos** que desean realizar análisis estadísticos sin necesidad de programar.

---

## 🚀 Funcionalidades

### 📂 Gestión de datos
- Carga de archivos **CSV, XLS, XLSX**
- Pegado directo de tablas desde **Excel / Google Sheets**
- Visualización tabular del dataset
- Limpieza del dataset en memoria
- Guardado del dataset actual en Excel
- Exportación de resultados

### 📈 Estadística descriptiva
- Media
- Mediana
- Desviación estándar
- Varianza
- Rango
- Coeficiente de variación
- Asimetría
- Curtosis
- Tablas de frecuencia para variables categóricas

### 🔗 Correlación y regresión
- Correlación **Pearson**
- Correlación **Spearman**
- Matriz de correlación con visualización gráfica
- Regresión lineal múltiple usando **OLS (statsmodels)**  
  Modelo:
  \[
  Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots
  \]

### 🧪 Pruebas estadísticas
- **t-test** para dos muestras independientes
- **Chi-cuadrado de independencia**
- **ANOVA** de un factor

### 📊 Gráficos
- Histograma
- Boxplot
- Scatter plot con línea de regresión
- Gráfico de barras
- Gráfico circular (pie)
- Distribución **Binomial**
- Distribución **Poisson**
- Distribución **Normal** con área sombreada
- Exportación de gráficos en formato PNG

### 📄 Exportación
- Exportación del análisis estadístico a archivo **TXT**
- Exportación del dataset a **Excel**
- Guardado de gráficos generados

---

## 🖥️ Requisitos del sistema

- **Python 3.9 o superior**
- Windows, Linux o macOS

> Tkinter viene incluido por defecto con Python.

---

## 📦 Dependencias

El proyecto utiliza las siguientes librerías de Python:

- pandas
- numpy
- matplotlib
- scipy
- statsmodels

---

## Clonar Repositorio
- git clone https://github.com/Samirssj/Estadisticas-con-Python.git
- cd Estadisticas-con-Python.git

## Instalar Dependencias 
- pip install pandas numpy matplotlib openpyxl scipy statsmodels python-dateutil

si no tienes SPSS te puede ayudar mucho ademas que es más facil de entender jejeje
## Ejecutar el programa en Windows🪟
- cd "ruta donde clonaste el repo"
- python graficos_estadisticos.py

## Ejecutar en linux 🐧
primero actualizamos nuestro linux con
- sudo apt update && apt upgrade -y
luego de eso instalamos las depencias y herramientas
- sudo apt install python3-venv python3-pip
Creamos un entorno virtual
- python -m venv estadistica
Una vez creado entramos al entorno
- source estadistica/bin/activate
Nuestra terminal nos mostrara algo haci
##┌──(estadistica)─(mvlwvr3㉿craxker)-[~/Estadisticas-con-Python]
##└─$
Eso quiere decir que ya estamos dentro del entorno virtual e instalamos los paquetes en caso no los tengan
-  pip install pandas numpy matplotlib openpyxl scipy statsmodels python-dateutil
Luego de eso ejecutamos el programa
- python graficos_estadisticos.py 
