# 02 - Generación y selección de textos

Este directorio contiene los notebooks encargados de generar abstracts académicos con el modelo ajustado y seleccionar los textos que serán utilizados en la siguiente iteración de reentrenamiento.

Esta fase representa el núcleo del proceso iterativo del proyecto. A partir de un modelo ya ajustado, se generan nuevos textos, se evalúan mediante métricas estadísticas y se seleccionan aquellos que presentan mayor cercanía al comportamiento humano.

## Archivos incluidos

* `02_generation_selection_manual.py`: estrategia basada en rangos manuales.
* `02_generation_selection_percentiles.py`: estrategia basada en percentiles dinámicos.

Ambos notebooks comparten la misma estructura general. La principal diferencia está en el criterio utilizado para filtrar y seleccionar los textos generados.

## Objetivo

El objetivo de esta fase es generar textos académicos y construir un subconjunto de textos seleccionados para reentrenar el modelo en la siguiente iteración.

Para ello, cada texto generado se evalúa con distintas métricas:

* Perplexity.
* GLTR mean rank.
* Repetición de bigramas.
* Presencia de artefactos LaTeX.
* Detector RoBERTa de Hello-SimpleAI.
* Distancia estadística respecto al conjunto humano.

## Flujo general

1. Carga del modelo base `mistralai/Mistral-7B-v0.1`.
2. Carga del adaptador LoRA correspondiente a la iteración seleccionada.
3. Generación de abstracts académicos mediante prompts controlados.
4. Cálculo de métricas de detectabilidad y calidad.
5. Aplicación de filtros básicos.
6. Selección de textos según la estrategia correspondiente.
7. Validación estadística mediante Cohen's d y Kolmogorov-Smirnov.
8. Exportación de los textos seleccionados en formato `.jsonl`.

## Estrategia manual

La estrategia manual utiliza rangos definidos a partir del análisis exploratorio de los resultados obtenidos.

En esta versión, se aplican límites fijos sobre métricas como Perplexity y GLTR:

```python
filtered = filtered[filtered["perplexity"].between(30, 60)]
filtered = filtered[filtered["gltr_mean_rank"] >= 120]
```

Posteriormente, los textos filtrados se ponderan según su distancia estadística al conjunto humano. Los textos más cercanos tienen mayor probabilidad de ser seleccionados, pero se mantiene cierta diversidad en la muestra final.

Esta estrategia permite controlar de forma más directa el comportamiento del filtrado, aunque depende de umbrales definidos manualmente.

## Estrategia por percentiles

La estrategia basada en percentiles no utiliza límites fijos. En su lugar, calcula los rangos de selección a partir de la distribución de los textos generados en cada iteración.

Ejemplo simplificado:

```python
ppl_low = filtered["perplexity"].quantile(0.25)
ppl_high = filtered["perplexity"].quantile(0.95)
gltr_low = filtered["gltr_mean_rank"].quantile(0.40)
gltr_high = filtered["gltr_mean_rank"].quantile(0.98)
```

Este enfoque permite que los límites se adapten dinámicamente a cada iteración. Su ventaja principal es que reduce la dependencia de valores manuales, aunque puede producir una evolución más suave y menos ajustada al conjunto humano.

## Diferencia principal entre ambas estrategias

| Estrategia  | Criterio de selección                             | Ventaja                                      | Limitación                             |
| ----------- | ------------------------------------------------- | -------------------------------------------- | -------------------------------------- |
| Manual      | Rangos fijos definidos tras análisis exploratorio | Mayor control sobre los textos seleccionados | Depende de umbrales manuales           |
| Percentiles | Rangos dinámicos calculados por distribución      | Más flexible y automática                    | Puede alejarse más del objetivo humano |

## Métrica de cercanía al conjunto humano

En ambas estrategias se calcula una distancia estadística respecto al conjunto humano usando Perplexity y GLTR.

Primero se normalizan ambas métricas mediante z-score:

```python
z_ppl = (perplexity - mean_ppl) / std_ppl
z_gltr = (gltr_mean_rank - mean_gltr) / std_gltr
```

Después se calcula una distancia euclídea normalizada:

```python
human_distance = sqrt(z_ppl² + z_gltr²)
```

Cuanto menor es esta distancia, mayor es la cercanía estadística del texto generado al conjunto humano.

## Gráficas generadas

Los notebooks generan gráficas de apoyo para analizar la distribución de las métricas antes y después de la selección.

Entre ellas:

* Histograma de `GLTR mean rank` en todos los textos generados.
* Histograma de `GLTR mean rank` en los textos seleccionados.
* Gráfica del impacto de cada filtro.
* Resumen estadístico de Perplexity y GLTR.

Estas gráficas permiten comprobar si la selección reduce el ruido y aproxima los textos generados al comportamiento humano.

## Salidas generadas

Cada notebook puede generar los siguientes archivos:

* CSV con todos los textos generados.
* CSV con métricas calculadas.
* CSV con textos seleccionados.
* JSONL preparado para el reentrenamiento de la siguiente iteración.

Los archivos generados no se incluyen en el repositorio debido a su tamaño y a que dependen de la ejecución experimental.

## Relación con el proyecto

Esta fase conecta el fine-tuning inicial con el reentrenamiento iterativo. Los textos seleccionados en esta etapa se utilizan posteriormente para construir un nuevo conjunto mixto junto con textos humanos, permitiendo repetir el ciclo de entrenamiento, generación, selección y evaluación.

La comparación entre ambas estrategias permite analizar cómo influye el criterio de selección en la evolución de la detectabilidad automática.
