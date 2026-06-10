# 02 - Generación y selección de textos

Este directorio contiene los notebooks encargados de generar abstracts académicos con el modelo ajustado y seleccionar los textos que serán utilizados en la siguiente iteración de reentrenamiento.

Esta fase representa el núcleo del proceso iterativo del proyecto: a partir de un modelo ya ajustado, se generan nuevos textos, se evalúan mediante métricas estadísticas y se seleccionan aquellos que presentan mayor cercanía al comportamiento humano.

## Archivos incluidos

- `02_generation_selection_manual.py`: estrategia basada en rangos manuales.
- `02_generation_selection_percentiles.py`: estrategia basada en percentiles dinámicos.

Ambos notebooks comparten la misma estructura general. La principal diferencia está en el criterio utilizado para filtrar y seleccionar los textos generados.

## Objetivo

El objetivo de esta fase es generar textos académicos y construir un subconjunto de textos seleccionados para reentrenar el modelo en la siguiente iteración.

Para ello, cada texto generado se evalúa con distintas métricas:

- Perplexity.
- GLTR mean rank.
- Repetición de bigramas.
- Presencia de artefactos LaTeX.
- Detector RoBERTa de Hello-SimpleAI.
- Distancia estadística respecto al conjunto humano.

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
