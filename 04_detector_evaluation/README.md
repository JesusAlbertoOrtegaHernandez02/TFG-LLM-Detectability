# 04 - Evaluación con métricas y detectores

Este directorio contiene el notebook encargado de evaluar los textos generados en las distintas iteraciones del modelo.

A diferencia de los notebooks anteriores, esta fase no entrena ni genera nuevos textos. Su objetivo es analizar los resultados obtenidos mediante métricas estadísticas y detectores automáticos, comparando el comportamiento de cada iteración con un conjunto humano de referencia.

## Archivo incluido

* `04_detector_evaluation.py`: notebook de evaluación final con métricas estadísticas, detectores automáticos y gráficas comparativas.

## Objetivo

El objetivo de este notebook es estudiar cómo evoluciona la detectabilidad automática de los textos generados a lo largo del proceso iterativo.

Para ello, se comparan textos humanos con textos generados por las distintas iteraciones del modelo:

```text
human, iter0, iter1, iter2, iter3, iter4, iter5
```

La evaluación combina métricas estadísticas internas con detectores externos, evitando depender de un único criterio.

## Métricas utilizadas

El notebook calcula y resume las siguientes métricas:

* Perplexity.
* GLTR mean rank.
* Número medio de palabras.
* Probabilidad de IA según Hello-SimpleAI.
* Probabilidad de IA según RADAR.
* Distancia estadística al conjunto humano.
* Cohen's d.
* Kolmogorov-Smirnov.
* Burstiness.

Estas métricas permiten analizar tanto la fluidez del texto como su predictibilidad, su cercanía estadística al conjunto humano y la respuesta de detectores automáticos.

## Detectores utilizados

### Hello-SimpleAI

Se utiliza el modelo:

```text
Hello-SimpleAI/chatgpt-detector-roberta
```

Este detector actúa como clasificador genérico de texto humano frente a texto generado por IA.

### RADAR

También se utiliza el detector:

```text
TrustSafeAI/RADAR-Vicuna-7B
```

RADAR se emplea como detector complementario, especialmente útil para analizar diferencias relativas entre iteraciones, aunque puede presentar desajustes de dominio en textos académicos.

## Flujo general

1. Carga del conjunto humano de referencia.
2. Carga de los textos generados por cada iteración.
3. Cálculo de Perplexity mediante GPT-2.
4. Cálculo de GLTR mean rank mediante GPT-2.
5. Cálculo de puntuaciones de detectores automáticos.
6. Construcción de una tabla resumen por iteración.
7. Comparación estadística entre humanos y modelos.
8. Generación de gráficas comparativas.
9. Exportación de tablas de métricas a CSV.

## Tabla resumen

El notebook genera una tabla final con los valores medios, desviaciones típicas y percentiles principales de cada métrica.

Esta tabla se guarda en:

```text
final_summary_table.csv
```

Entre otros campos, incluye:

* media y desviación típica de Perplexity,
* media y desviación típica de GLTR,
* percentiles 10, 50 y 90,
* puntuaciones medias de detectores,
* distancia media al comportamiento humano,
* porcentaje de textos dentro de rangos considerados cercanos al humano.

## Comparación estadística

Además de las medias, el notebook compara las distribuciones humanas y generadas mediante:

* Cohen's d,
* test de Kolmogorov-Smirnov.

Esto permite medir no solo si las medias se aproximan, sino también si las distribuciones completas siguen siendo distinguibles.

La tabla resultante se guarda como:

```text
human_vs_models_separation.csv
```

## Gráficas generadas

El notebook genera varias gráficas de apoyo para interpretar los resultados.

Entre ellas:

* Evolución de Perplexity y GLTR por iteración.
* Comparativa de detectores automáticos.
* Distribución de distancia al comportamiento humano.
* Matriz de confusión para RADAR.
* Comparación visual entre humanos y modelos en el espacio Perplexity-GLTR.
* Evolución de la distancia ponderada al conjunto humano.
* Distribución de GLTR entre humanos y la mejor iteración.
* Burstiness por modelo.

Estas gráficas se utilizan para analizar si el modelo se aproxima progresivamente al comportamiento humano y si esa aproximación reduce la detectabilidad automática.

## Interpretación de las gráficas principales

### Evolución de Perplexity y GLTR

Esta gráfica permite observar cómo cambian dos métricas fundamentales a lo largo de las iteraciones.

Perplexity mide la fluidez probabilística del texto, mientras que GLTR mean rank refleja la predictibilidad de los tokens generados.

Una buena iteración no debe optimizar únicamente una métrica aislada, sino buscar un equilibrio entre ambas.

### Comparativa de detectores

La gráfica de detectores compara la probabilidad media de clasificación como IA según Hello-SimpleAI y RADAR.

Esta comparación muestra que los detectores pueden comportarse de forma distinta según el dominio del texto. Por ello, el proyecto no basa sus conclusiones únicamente en la salida de un detector, sino en la combinación de varias métricas.

### Distancia al conjunto humano

La distancia al comportamiento humano se calcula usando principalmente Perplexity y GLTR.

Esta métrica resume la cercanía estadística de cada iteración respecto al conjunto humano de referencia. Una distancia menor indica una mayor aproximación al comportamiento humano.

### Espacio Perplexity-GLTR

El notebook también genera gráficas de dispersión donde cada punto representa un texto.

En estas gráficas, el eje X corresponde a Perplexity y el eje Y a GLTR mean rank. Esto permite comparar visualmente si la nube de puntos de una iteración se aproxima a la nube de textos humanos.

## Salidas generadas

El notebook genera principalmente:

* `final_summary_table.csv`
* `human_vs_models_separation.csv`
* gráficas comparativas de métricas y detectores.

Estos archivos no se incluyen necesariamente en el repositorio porque dependen de la ejecución experimental y pueden ocupar espacio adicional.

## Relación con el proyecto

Este notebook corresponde a la fase final del pipeline experimental:

1. fine-tuning inicial,
2. generación y selección de textos,
3. reentrenamiento iterativo,
4. evaluación mediante métricas y detectores.

Su función es proporcionar una visión global de la evolución del modelo y permitir comparar de forma objetiva las distintas iteraciones.

## Nota de reproducibilidad

Para ejecutar este notebook es necesario disponer de:

* los CSV con textos generados por iteración,
* el conjunto humano de referencia,
* acceso a los modelos usados para calcular Perplexity, GLTR y detectores,
* una GPU compatible si se desea reducir el tiempo de cálculo.

Los resultados pueden variar ligeramente dependiendo del entorno de ejecución y de las versiones concretas de las librerías utilizadas.
