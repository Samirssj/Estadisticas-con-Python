# Manual de uso: Estadística Web

Guía completa para analizar datos en el navegador, sin instalar SPSS, Python ni ningún programa. Sirve para estudiantes, docentes y analistas que necesitan estadística descriptiva e inferencial, gráficos y modelos básicos a partir de un archivo de Excel o CSV.

---

## 1. Qué es y qué puede hacer

Estadística Web es una página que calcula todo dentro de tu navegador. Los datos que cargas **no se envían a ningún servidor**.

| Sección | Qué hace |
|---|---|
| **Datos** | Carga de archivos, pegado desde Excel, vista de datos y vista de variables, exportación. |
| **Descriptiva** | Media, mediana, desviación estándar, cuartiles, asimetría, curtosis y tablas de frecuencia. |
| **Correlación** | Pearson y Spearman con valores p y mapa de calor. |
| **Regresión** | Regresión lineal múltiple (OLS) con variables numéricas y categóricas. |
| **Pruebas** | t de Student (2 muestras), chi-cuadrado de independencia y ANOVA de un factor. |
| **Gráficos** | Histograma, boxplot, dispersión, barras y circular. |
| **Distribuciones** | Binomial, Poisson y Normal con cálculo de probabilidades y áreas sombreadas. |

**Requisitos:** un navegador actual (Chrome, Edge, Firefox o Safari) y conexión a internet para abrir la página. No hace falta crear cuenta.

---

## 2. Primeros pasos en dos minutos

1. Abre la página. Viene cargada con **datos de ejemplo** (90 estudiantes con nota final en escala 0 a 20), así puedes probar todo de inmediato.
2. Entra a la pestaña **Descriptiva**: verás el resumen de cada variable numérica.
3. Entra a **Gráficos** y pulsa **Histograma**.
4. Cuando quieras usar tus datos, vuelve a **Datos** y pulsa **Elegir archivo** (o arrastra tu archivo al recuadro).

> Todos los resultados se actualizan al cargar un archivo nuevo. Los resultados anteriores (regresión, pruebas) se borran para que no se mezclen con datos distintos.

---

## 3. Preparar tus datos

La página lee una tabla con esta forma:

| edad | sexo | horas_estudio | nota_final |
|---|---|---|---|
| 21 | F | 7.1 | 12.4 |
| 19 | M | 4.3 | 8.0 |

Reglas importantes:

- **La primera fila son los nombres de las columnas.** Cada fila siguiente es un caso (una persona, una medición, etc.).
- **Una columna, una variable.** No mezcles números y texto en la misma columna si quieres que se trate como numérica.
- **Nombres de columna únicos.** Si hay nombres repetidos, la página agrega `_2`, `_3`. Si un encabezado está vacío, lo llama `columna_1`, `columna_2`, etc.
- **No dejes filas de totales ni notas** debajo de la tabla: se leerían como datos.
- **Valores perdidos:** una celda vacía, o con `NA`, `N/A`, `NaN`, `nan`, `null`, `NULL`, `None` o `#N/A`, se cuenta como perdida.
- **Decimales:** usa punto (`7.5`). Si tu archivo usa `;` o tabulaciones como separador, también se aceptan decimales con coma (`7,5`). En un CSV separado por comas, la coma es el separador y no se interpreta como decimal.

### Cómo decide si una variable es numérica o categórica

- Si **todos** los valores no perdidos son números, la variable es **numérica**.
- Si hay aunque sea un valor que no es número, la variable es **categórica** y todos sus valores se tratan como texto.

Un código como `1` y `2` para sexo se tratará como numérico. Para usarlo como grupo en una prueba o un gráfico de barras no hay problema, porque esas secciones permiten elegir cualquier columna. Si prefieres que sea categórica en la regresión, escribe `M` y `F` o `Hombre` y `Mujer` en tu archivo.

---

## 4. Sección Datos

### 4.1 Formas de cargar datos

| Forma | Cómo se hace |
|---|---|
| **Elegir archivo** | Botón azul. Acepta `.csv`, `.tsv`, `.txt`, `.xls`, `.xlsx`, `.xlsm` y `.ods`. |
| **Arrastrar y soltar** | Arrastra el archivo al recuadro punteado. |
| **Pegar desde Excel o Google Sheets** | Copia las celdas (incluida la fila de encabezados) y pulsa **Ctrl+V** con la página abierta, sin tener el cursor dentro de un cuadro de texto. |
| **Cuadro de texto** | Abre "Pegar texto de una tabla", pega el contenido y pulsa **Usar tabla pegada**. |
| **Datos de ejemplo** | Botón gris. Carga el conjunto de 90 estudiantes. |

El separador de columnas (coma, punto y coma o tabulación) se detecta solo.

**Archivos de Excel con varias hojas:** aparece un selector **Hoja** sobre la tabla. Al cambiarlo se carga esa hoja.

**Si falla la carga:** ve a la sección 13 (Problemas frecuentes).

### 4.2 Vista de datos

Muestra la tabla con las primeras **500 filas** (los análisis usan todas). Encima de la tabla aparecen "chips" con cada columna: su nombre, su tipo (`num` o `cat`) y cuántos valores vacíos tiene.

### 4.3 Vista de variables

Pulsa **Vista de variables** para ver, como en SPSS, una fila por variable con:

- Nombre
- Tipo (numérica o categórica)
- Válidos y perdidos
- Valores distintos
- Rango (para numéricas) o primeras categorías (para categóricas)

Úsala para revisar que cada columna se interpretó como esperabas antes de analizar.

### 4.4 Botones de la barra de datos

| Botón | Qué hace |
|---|---|
| **Quitar filas con vacíos** | Elimina de la memoria todas las filas que tengan algún valor perdido. Tu archivo original no cambia. |
| **Exportar Excel** | Descarga la tabla actual como `.xlsx`. |
| **Exportar CSV** | Descarga la tabla actual como `.csv` (con acentos compatibles con Excel). |
| **Informe TXT** | Genera un informe de texto con descriptivos, frecuencias y los últimos resultados de correlación, regresión y pruebas que hayas ejecutado. |
| **Limpiar datos** | Vacía la tabla para empezar de cero. |

> Al exportar, el visor te muestra un cuadro de confirmación con el nombre del archivo. Acéptalo para guardarlo.

---

## 5. Sección Descriptiva

Se calcula sola al cargar los datos.

### 5.1 Variables numéricas

Una fila por variable, con estas columnas:

| Columna | Significado |
|---|---|
| **n** | Valores válidos. |
| **Faltantes** | Valores perdidos. |
| **Media** | Promedio. |
| **Desv. est.** | Desviación estándar muestral (divide entre n−1). |
| **Mín., Q1, Mediana, Q3, Máx.** | Mínimo, cuartiles y máximo. Los cuartiles usan interpolación lineal (igual que pandas y R tipo 7). |
| **Rango** | Máx. − Mín. |
| **Varianza** | Desviación estándar al cuadrado. |
| **CV** | Coeficiente de variación = desviación estándar / valor absoluto de la media. Aparece "—" si la media es 0. |
| **Asimetría** | Positiva: cola larga a la derecha. Negativa: a la izquierda. Cerca de 0: simétrica. |
| **Curtosis** | Exceso de curtosis. Cerca de 0 se parece a la normal; positiva indica colas más pesadas. |

Asimetría y curtosis necesitan al menos 3 y 4 datos respectivamente. Si no, aparece "—".

El botón **Copiar como texto** copia la tabla al portapapeles para pegarla en Word o en un correo.

### 5.2 Tablas de frecuencia

Una tabla por variable categórica con **Frecuencia**, **%** y **% acumulado**. El selector **Variable** permite ver una sola categórica o todas a la vez. Si una variable tiene más de 50 categorías, se muestran las 50 más frecuentes. Los valores perdidos aparecen como `(vacío)`.

---

## 6. Sección Correlación

La correlación mide qué tan fuerte es la relación entre dos variables numéricas, de −1 a +1.

### 6.1 Cómo usarla

1. Elige el **Método**:
   - **Pearson:** relación lineal. Úsalo cuando los datos son aproximadamente normales y sin valores extremos.
   - **Spearman:** relación monótona basada en rangos. Úsalo con datos ordinales, sesgados o con valores atípicos.
2. Marca las **variables numéricas** a incluir (por defecto, las primeras 10). Necesitas al menos 2.
3. Pulsa **Calcular** (también se recalcula al cambiar el método).

### 6.2 Qué verás

- **Matriz de coeficientes**, coloreada: azul para correlación positiva y naranja para negativa. Mientras más intenso el color, más fuerte la relación.
- **Matriz de valores p** (dos colas). Un valor menor que tu nivel α (normalmente 0.05) indica una correlación significativa.
- **Observaciones por par** (en el desplegable): cada par usa solo las filas donde ambas variables tienen valor.
- **Mapa de calor** con la misma información. Pasa el cursor sobre una celda para ver el valor exacto. **Guardar PNG** descarga la imagen.

### 6.3 Cómo interpretar r

| |r| | Lectura habitual |
|---|---|
| 0.00 a 0.19 | Muy débil |
| 0.20 a 0.39 | Débil |
| 0.40 a 0.59 | Moderada |
| 0.60 a 0.79 | Fuerte |
| 0.80 a 1.00 | Muy fuerte |

> Correlación no es causalidad. Que dos variables se muevan juntas no demuestra que una cause la otra.

**Ejemplo con los datos de ejemplo:** `horas_estudio` y `nota_final` tienen r = 0.7095 (p < 0.0001): relación fuerte y positiva. Quien estudia más horas tiende a sacar mejor nota.

---

## 7. Sección Regresión

Ajusta el modelo **Y = β₀ + β₁X₁ + β₂X₂ + …** por mínimos cuadrados ordinarios (OLS).

### 7.1 Cómo usarla

1. En **Variable dependiente (Y)** elige la variable que quieres explicar (debe ser numérica).
2. Marca las **variables independientes (X)**. Pueden ser numéricas o categóricas. Por defecto se marcan dos numéricas.
3. Pulsa **Ejecutar regresión**.

**Variables categóricas:** se convierten automáticamente en variables indicadoras (0 o 1). La **primera categoría en orden alfabético es la referencia**. Por ejemplo, con `carrera` (Contabilidad, Derecho, Sistemas) aparecen los coeficientes `carrera[Derecho]` y `carrera[Sistemas]`, que se leen como diferencia frente a Contabilidad. Una categórica puede tener hasta 20 categorías.

**Filas usadas:** solo las que tienen valor en Y y en todas las X. La página indica cuántas filas se excluyeron.

### 7.2 Resultados

**Resumen del modelo**

| Indicador | Qué indica |
|---|---|
| **R²** | Proporción de la variación de Y que explica el modelo (0 a 1). |
| **R² ajustado** | R² corregido por el número de variables. Útil para comparar modelos. |
| **F y Valor p (F)** | Prueba global: ¿el modelo explica algo? Un p pequeño dice que sí. |
| **Observaciones** | Filas usadas. |
| **Error residual** | Desviación típica de los errores, en las unidades de Y. |
| **AIC y BIC** | Criterios para comparar modelos: menores es mejor. |

Debajo aparece la **ecuación estimada**.

**Tabla de coeficientes**

| Columna | Significado |
|---|---|
| **Coeficiente** | Cambio esperado en Y por cada unidad que sube esa X, manteniendo las demás constantes. |
| **Error est.** | Incertidumbre del coeficiente. |
| **t y P>\|t\|** | Prueba de si el coeficiente es distinto de 0. |
| **IC 95%** | Intervalo de confianza del coeficiente. |
| **Sig.** | `***` p<0.001, `**` p<0.01, `*` p<0.05, `.` p<0.1. |

Al final se resume en una frase cuánto explica el modelo y si es significativo en conjunto.

### 7.3 Gráficos de diagnóstico

- **Observado vs. predicho:** los puntos deben acercarse a la línea punteada.
- **Residuos vs. predicho:** deben repartirse al azar alrededor de 0, sin forma de embudo ni de curva. Si ves un patrón, el modelo puede estar mal especificado.

### 7.4 Errores frecuentes

| Mensaje | Qué hacer |
|---|---|
| "Las variables independientes son linealmente dependientes" | Quita una variable que sea combinación de otras (por ejemplo, una columna constante o dos columnas idénticas). |
| "Hacen falta más observaciones completas que parámetros" | Hay muy pocas filas completas. Quita variables o completa datos. |
| "tiene una sola categoría" | Esa X no varía; quítala. |
| "tiene N categorías; usa una variable con 20 o menos" | Agrupa categorías en tu archivo. |

**Ejemplo con los datos de ejemplo:** `nota_final ~ carrera + horas_estudio + asistencia + trabaja` da R² = 0.6489 y R² ajustado = 0.6267 con 85 observaciones. Cada hora de estudio adicional suma unos 0.48 puntos a la nota (p < 0.0001), y `carrera[Sistemas]` suma 1.24 puntos frente a Contabilidad (p = 0.0038).

---

## 8. Sección Pruebas

Antes de ejecutar, elige el **nivel de significancia (α)**: 0.10, 0.05 (por defecto) o 0.01. Cada prueba concluye comparando su valor p con α:

- **p < α:** se rechaza H₀ (hay evidencia de diferencia o asociación).
- **p ≥ α:** no se rechaza H₀ (no hay evidencia suficiente).

Cada prueba ignora las filas con valores perdidos en las variables elegidas.

### 8.1 t de Student para 2 muestras

Compara las medias de una variable numérica entre dos grupos.

**Pasos:**
1. **Variable numérica:** lo que se mide (por ejemplo, `nota_final`).
2. **Variable de grupo:** la columna que define los grupos (por ejemplo, `trabaja`).
3. **Grupo A** y **Grupo B:** los dos valores a comparar. Se llenan solos con las categorías de la variable; elige cuáles.
4. **Varianzas:**
   - **Distintas (Welch):** recomendada por defecto, es segura aunque los grupos tengan dispersión diferente.
   - **Iguales (Student):** úsala solo si sabes que las varianzas son parecidas.
5. Pulsa **Ejecutar t-test**.

**Resultado:** tabla con n, media y desviación por grupo; luego t, grados de libertad, valor p, diferencia de medias (A − B), su intervalo de confianza y la **d de Cohen** (tamaño del efecto: 0.2 pequeño, 0.5 mediano, 0.8 grande).

**H₀:** las medias de los dos grupos son iguales.

**Ejemplo:** `nota_final` por `trabaja` (No vs. Sí): t = 0.2477, p = 0.8053. Con α = 0.05 no se rechaza H₀: no hay evidencia de que trabajar cambie la nota media.

### 8.2 Chi-cuadrado de independencia

Contrasta si dos variables categóricas están asociadas.

**Pasos:** elige **Variable 1** y **Variable 2** (distintas, con 30 categorías o menos) y pulsa **Ejecutar chi²**.

**Resultado:** tabla de frecuencias observadas con totales; χ², grados de libertad, valor p, **V de Cramér** (fuerza de la asociación de 0 a 1) y las frecuencias esperadas en un desplegable.

**Notas:**
- En tablas 2×2 se aplica la **corrección de Yates**, igual que scipy.
- Si hay casillas con frecuencia esperada menor que 5, aparece un aviso: el resultado puede ser poco fiable. Agrupa categorías si puedes.

**H₀:** las variables son independientes.

**Ejemplo:** `sexo` y `trabaja`: χ² = 0.0012, p = 0.9726. No hay evidencia de asociación.

### 8.3 ANOVA de un factor

Compara las medias de una variable numérica entre **dos o más grupos**.

**Pasos:** elige la **Variable numérica** y el **Factor (grupos)** (hasta 30 grupos) y pulsa **Ejecutar ANOVA**.

**Resultado:** tabla de ANOVA (suma de cuadrados, gl, media cuadrática, F, valor p), **η²** (proporción de variación explicada por el factor) y una tabla con n, media y desviación de cada grupo.

**H₀:** todas las medias de grupo son iguales.

**Ejemplo:** `nota_final` por `carrera`: F = 5.2560, p = 0.0070, η² = 0.1078. Con α = 0.05 se rechaza H₀: al menos una carrera tiene una media distinta. Las medias son 9.35 (Contabilidad), 8.59 (Derecho) y 10.61 (Sistemas).

> El ANOVA dice que *alguna* media difiere, pero no cuál. Mira la tabla de medias por grupo para ver la dirección.

### 8.4 Cuál prueba usar

| Pregunta | Prueba |
|---|---|
| ¿Dos grupos tienen distinta media? | t de Student |
| ¿Tres o más grupos tienen distinta media? | ANOVA |
| ¿Dos variables categóricas están relacionadas? | Chi-cuadrado |
| ¿Dos variables numéricas se relacionan? | Correlación |
| ¿Cuánto influyen varias variables sobre una numérica? | Regresión |

---

## 9. Sección Gráficos

Elige las variables arriba y pulsa el gráfico que quieras. El gráfico aparece debajo.

| Control | Uso |
|---|---|
| **Variable 1 (X)** | Variable principal. |
| **Variable 2 (Y o grupo)** | Segunda variable (dispersión) o variable de grupo (boxplot). |
| **Intervalos** | Número de barras del histograma (1 a 200, por defecto 12). |
| **Curva normal** | Superpone la curva normal con la media y desviación de los datos al histograma. |

| Gráfico | Qué usa | Detalles |
|---|---|---|
| **Histograma** | Variable 1 numérica | Muestra n, media y desviación en el subtítulo. Pasa el cursor sobre una barra para ver su rango y frecuencia. |
| **Boxplot** | Variable 1 numérica; opcional Variable 2 como grupo | Caja de Q1 a Q3, línea en la mediana, bigotes hasta 1.5 veces el rango intercuartílico y círculos para valores atípicos. Con variable 2 dibuja una caja por grupo (máximo 30). |
| **Dispersión X vs Y** | Variables 1 y 2 numéricas | Dibuja los puntos y la recta de regresión, con ecuación, r, R² y n. Si hay más de 4 000 filas se dibujan 4 000 puntos para que sea fluido; la recta y los estadísticos usan todos los datos. |
| **Barras** | Variable 1 | Frecuencia de cada categoría. Muestra las 30 más frecuentes. |
| **Circular** | Variable 1 | Porcentaje por categoría. Muestra las 7 más frecuentes y agrupa el resto en "Otras". |

**Guardar PNG** descarga la imagen del gráfico actual.

Consejos:
- Los colores siguen un orden fijo y están elegidos para distinguirse también con daltonismo.
- La página respeta el modo claro u oscuro de tu dispositivo.

---

## 10. Sección Distribuciones

Esta sección no usa tus datos (salvo la opción de la Normal): sirve para calcular y visualizar probabilidades teóricas.

### 10.1 Binomial

Número de éxitos en **n** ensayos independientes, cada uno con probabilidad de éxito **p**.

- **n:** entero de 1 a 1000.
- **p:** entre 0 y 1.
- **P(a ≤ X ≤ b):** opcional. Si completas **a** y/o **b**, esas barras se colorean y se calcula la probabilidad del rango. Deja uno vacío para "hasta b" o "desde a".

Ejemplo: n = 20, p = 0.4, a = 5, b = 10 da P(5 ≤ X ≤ 10) = 0.82153. Debajo del gráfico verás la media (np) y la desviación estándar.

### 10.2 Poisson

Número de eventos en un intervalo, con media **λ**.

- **λ:** mayor que 0 y hasta 1000.
- **a** y **b:** igual que en la binomial.

### 10.3 Normal

- **Tomar media y desviación de una variable:** elige una variable numérica de tus datos y se rellenan **Media** y **Desv. est.** con sus valores. O deja la lista vacía y escribe tus propios valores.
- **Área desde a hasta b:** sombrea el área bajo la curva y calcula la probabilidad. Deja **a** vacío para "desde −∞" o **b** vacío para "hasta +∞".

Ejemplo: media 12, desviación 3, a = 10, b = 14 da P(10 < X < 14) = 0.49501 (49.50 %).

La desviación estándar debe ser mayor que 0 y, si das los dos límites, a debe ser menor que b.

---

## 11. Exportar y guardar resultados

| Qué | Dónde | Formato |
|---|---|---|
| Tabla de datos | Datos → Exportar Excel / Exportar CSV | `.xlsx`, `.csv` |
| Informe de análisis | Datos → Informe TXT | `.txt` |
| Tabla descriptiva | Descriptiva → Copiar como texto | Portapapeles |
| Cualquier gráfico | Botón **Guardar PNG** bajo el gráfico | `.png` |

**Qué incluye el Informe TXT:** el resumen de variables numéricas, las frecuencias de las categóricas y los resultados de **correlación, regresión y pruebas que hayas ejecutado en esta sesión**. Ejecuta primero lo que quieras que aparezca.

---

## 12. Un flujo de trabajo completo (ejemplo)

Objetivo: saber si las horas de estudio y la carrera influyen en la nota final.

1. **Datos →** Datos de ejemplo. Revisa la **Vista de variables**: dos columnas tienen valores perdidos (`horas_estudio` y `asistencia`).
2. **Descriptiva →** mira la media (9.52) y la desviación (2.49) de `nota_final`.
3. **Gráficos →** Variable 1 = `nota_final`, **Histograma** con **Curva normal** para ver si se parece a una normal.
4. **Gráficos →** Variable 1 = `nota_final`, Variable 2 = `carrera`, **Boxplot** para comparar carreras visualmente.
5. **Correlación →** Pearson. Ve que `horas_estudio` y `nota_final` se relacionan con r = 0.71.
6. **Pruebas → ANOVA:** `nota_final` por `carrera` da p = 0.0070. Hay diferencias entre carreras.
7. **Regresión →** Y = `nota_final`, X = `horas_estudio`, `asistencia`, `carrera`. Revisa R², los coeficientes y el gráfico de residuos.
8. **Datos → Informe TXT** para guardar todo.

---

## 13. Problemas frecuentes

| Problema | Causa probable | Solución |
|---|---|---|
| "El archivo necesita una fila de encabezados y al menos una fila de datos" | El archivo tiene solo una fila o está vacío. | Verifica que la primera fila sean nombres y que haya datos debajo. |
| Una columna numérica aparece como `cat` | Tiene algún valor de texto (una nota, `-`, un símbolo de moneda, etc.). | Corrige esa celda en tu archivo o déjala vacía. Revisa la Vista de variables. |
| Los decimales se leen como texto | Usas coma decimal en un CSV separado por comas. | Guarda el CSV con `;` como separador o cambia a punto decimal. |
| "No se pudo cargar el lector de Excel" | El componente de lectura de Excel no se descargó (conexión, bloqueo de red). | Guarda tu archivo como CSV y súbelo de nuevo. |
| Un `.xls` muy antiguo no abre | Formato dañado o protegido con contraseña. | Ábrelo en Excel y guárdalo como `.xlsx` o `.csv`. |
| Pegué una tabla y salió una sola columna | Se copió texto en lugar de celdas. | Copia las celdas directamente desde Excel o Sheets. |
| "Elige al menos 2 variables" en correlación | Hay menos de 2 columnas numéricas marcadas. | Marca más variables numéricas. |
| El gráfico no aparece | No elegiste las variables necesarias, o la variable no es numérica. | Lee el mensaje en rojo sobre el gráfico: indica qué falta. |
| El botón de exportar no descarga nada | El visor muestra primero un cuadro de confirmación. | Acepta el cuadro para guardar el archivo. |
| Un valor p aparece como `< 0.0001` | El valor real es menor que 0.0001. | Es normal: se interpreta como altamente significativo. |

Si ocurre un error inesperado, la página muestra un aviso rojo en la parte inferior con el detalle. Recargar la página y volver a cargar los datos suele resolverlo.

---

## 14. Límites de esta herramienta

- No modifica celdas ni recodifica variables: prepara tus datos en Excel antes de cargarlos.
- No incluye pruebas no paramétricas (Mann-Whitney, Kruskal-Wallis), regresión logística ni análisis factorial.
- No hace pruebas post hoc tras el ANOVA.
- El gráfico de dispersión dibuja como máximo 4 000 puntos.
- La regresión permite hasta 20 categorías por variable categórica, y chi-cuadrado y ANOVA hasta 30 categorías o grupos.
- Los resultados son una ayuda para estudiar y explorar. Para publicar o tomar decisiones críticas, verifica los supuestos de cada prueba y contrasta con un software estadístico de referencia.

Los cálculos se compararon con **scipy** y **statsmodels** (Python) sobre los datos de ejemplo y coinciden en descriptivos, correlación, regresión, t-test, chi-cuadrado, ANOVA y distribuciones.

---

## 15. Glosario rápido

| Término | Significado |
|---|---|
| **Valor p** | Probabilidad de ver un resultado tan extremo como el observado si H₀ fuera cierta. Pequeño = evidencia contra H₀. |
| **α (nivel de significancia)** | Umbral para decidir. Habitualmente 0.05. |
| **H₀** | Hipótesis nula: "no hay diferencia" o "no hay relación". |
| **Grados de libertad (gl)** | Cantidad de datos libres de variar al estimar. Dependen del tamaño de muestra. |
| **Intervalo de confianza (IC)** | Rango plausible para el valor real de un parámetro. |
| **Variable numérica** | Se mide con números (edad, nota, peso). |
| **Variable categórica** | Se clasifica en grupos (sexo, carrera). |
| **Outlier (valor atípico)** | Valor muy alejado del resto. |
| **Tamaño del efecto** | Qué tan grande es una diferencia, independientemente del tamaño de la muestra (d de Cohen, η², V de Cramér). |
| **Residuo** | Diferencia entre el valor observado y el que predice el modelo. |
