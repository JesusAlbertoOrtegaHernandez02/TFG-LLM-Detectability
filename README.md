# TFG-LLM-Detectability

Repositorio asociado al Trabajo de Fin de Grado:

**Estudio y ajuste de modelos de lenguaje generativos open-source para la generación de textos con baja detectabilidad automática**

El proyecto analiza de forma experimental cómo evoluciona la detectabilidad automática de textos generados por un modelo de lenguaje open-source tras sucesivos ciclos de ajuste fino, generación, selección y reentrenamiento.

El objetivo no es desarrollar un sistema de evasión, sino estudiar qué patrones estadísticos y lingüísticos cambian durante el proceso y cómo responden distintas métricas y detectores automáticos.

## Resumen del proyecto

El trabajo parte de un modelo generativo open-source, `mistralai/Mistral-7B-v0.1`, ajustado sobre abstracts académicos procedentes de ArXiv.

A partir de ese modelo inicial, se construye un pipeline iterativo compuesto por las siguientes fases:

1. ajuste fino inicial,
2. generación de textos académicos,
3. cálculo de métricas estadísticas y de detectabilidad,
4. selección de textos cercanos al comportamiento humano,
5. reentrenamiento del modelo,
6. evaluación comparativa entre iteraciones.

Durante el proceso se analizan métricas como Perplexity, GLTR, Burstiness, Cohen's d, Kolmogorov-Smirnov y la salida de detectores automáticos como Hello-SimpleAI y RADAR.

## Estructura del repositorio

```text
TFG-LLM-Detectability/
│
├── 01_initial_finetuning/
│   ├── README.md
│   └── 01_initial_finetuning.py
│
├── 02_generation_and_selection/
│   ├── README.md
│   ├── 02_generation_selection_manual.py
│   └── 02_generation_selection_percentiles.py
│
├── 03_iterative_retraining/
│   ├── README.md
│   └── 03_iterative_retraining.py
│
├── 04_detector_evaluation/
│   ├── README.md
│   └── 04_detector_evaluation.py
│
├── figures/
│
└── docs/
```

## Contenido de cada carpeta

### `01_initial_finetuning/`

Contiene el notebook correspondiente al primer ajuste fino del modelo.

En esta fase se carga el dataset académico, se limpian los abstracts, se prepara el conjunto de entrenamiento y se realiza el primer fine-tuning de `mistralai/Mistral-7B-v0.1` mediante LoRA y cuantización en 4 bits.

El resultado de esta fase es el adaptador inicial `iter0_academic`, que sirve como punto de partida para el resto del pipeline.

### `02_generation_and_selection/`

Contiene los notebooks encargados de generar textos académicos y seleccionar aquellos que serán utilizados en la siguiente iteración.

Incluye dos estrategias de selección:

* `02_generation_selection_manual.py`: utiliza rangos manuales definidos a partir del análisis exploratorio.
* `02_generation_selection_percentiles.py`: utiliza rangos dinámicos calculados mediante percentiles.

Ambas estrategias calculan métricas como Perplexity, GLTR, repetición de bigramas, puntuación de detector y distancia estadística respecto al conjunto humano.

### `03_iterative_retraining/`

Contiene el notebook de reentrenamiento iterativo.

En esta fase se combinan los textos generados seleccionados con textos humanos de referencia para construir un dataset mixto. Después se carga el adaptador LoRA de la iteración anterior y se continúa el ajuste fino para obtener una nueva iteración del modelo.

Esta fase permite repetir el ciclo generación-selección-reentrenamiento.

### `04_detector_evaluation/`

Contiene el notebook de evaluación final.

En esta fase se comparan los textos humanos y los textos generados por las distintas iteraciones mediante métricas estadísticas y detectores automáticos.

El notebook genera tablas resumen, comparaciones mediante Cohen's d y Kolmogorov-Smirnov, gráficas de evolución de métricas y análisis de detectabilidad por iteración.

## Pipeline experimental

El flujo general del proyecto puede resumirse así:

```text
Dataset académico
        ↓
Fine-tuning inicial con LoRA
        ↓
Generación de textos
        ↓
Cálculo de métricas
        ↓
Selección de textos cercanos al comportamiento humano
        ↓
Reentrenamiento iterativo
        ↓
Evaluación con métricas y detectores
```

## Modelos y herramientas principales

* `mistralai/Mistral-7B-v0.1`
* LoRA
* PEFT
* BitsAndBytes
* Hugging Face Transformers
* Hugging Face Datasets
* GPT-2 para Perplexity y GLTR
* Hello-SimpleAI RoBERTa detector
* RADAR detector
* Google Colab

## Nota sobre datasets y modelos

Los datasets generados, checkpoints, adaptadores LoRA y archivos de métricas completos no se incluyen directamente en el repositorio debido a su tamaño y a que dependen de la ejecución experimental.

El código mantiene explícitas las rutas y nombres de los archivos utilizados para facilitar la reproducibilidad del proceso.

## Finalidad académica

Este repositorio tiene una finalidad exclusivamente académica e investigadora.

El objetivo del trabajo es analizar la evolución de la detectabilidad automática de textos generados y estudiar las limitaciones de las métricas y detectores actuales, no proporcionar un sistema orientado a evadir herramientas de detección.
