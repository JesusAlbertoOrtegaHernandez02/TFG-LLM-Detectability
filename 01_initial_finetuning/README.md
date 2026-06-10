# 01 - Fine-tuning inicial

Este directorio contiene el primer notebook del proyecto, correspondiente a la preparación del dataset académico y al primer ajuste fino del modelo generativo seleccionado.

El objetivo principal de esta fase es obtener una primera versión ajustada del modelo, denominada `iter0_academic`, que servirá como punto de partida para las siguientes iteraciones de generación, filtrado y reentrenamiento.

## Objetivo del notebook

El notebook realiza el fine-tuning inicial de `mistralai/Mistral-7B-v0.1` sobre una muestra de abstracts académicos procedentes de ArXiv. Para hacer viable el entrenamiento en Google Colab, se emplean técnicas de cuantización y fine-tuning eficiente mediante LoRA.

## Fases principales

1. Instalación de dependencias necesarias.
2. Montaje de Google Drive.
3. Carga del dataset académico desde Hugging Face.
4. Limpieza básica de abstracts.
5. Filtrado por longitud.
6. Selección de una muestra de 10.000 textos.
7. División del dataset en train y test.
8. Comprobación de solapamiento entre train y test.
9. Carga de Mistral-7B con cuantización en 4 bits.
10. Configuración de LoRA.
11. Tokenización de los textos.
12. Entrenamiento inicial del modelo.
13. Visualización de la evolución de la pérdida.
14. Guardado del adaptador LoRA entrenado.
15. Generación de un ejemplo de texto.

## Modelo utilizado

- `mistralai/Mistral-7B-v0.1`

El modelo se carga con cuantización en 4 bits mediante `bitsandbytes`, reduciendo el consumo de memoria GPU y permitiendo su uso en un entorno limitado como Google Colab.

## Técnicas utilizadas

- Fine-tuning eficiente mediante LoRA.
- PEFT.
- Cuantización en 4 bits.
- Entrenamiento autoregresivo.
- Optimización con `paged_adamw_8bit`.
- División reproducible mediante `seed=42`.

## Dataset

Se utiliza el dataset:

- `CShorten/ML-ArXiv-Papers`

A partir de este conjunto se limpian los abstracts y se selecciona una muestra de 10.000 textos. Posteriormente, se divide el conjunto en entrenamiento y evaluación.

Los datos completos no se incluyen en el repositorio debido a su tamaño.

## Salidas generadas

El notebook genera:

- `train_split_iter0.json`
- `test_split_iter0.json`
- Adaptador LoRA de la primera iteración: `iter0_academic`
- Una generación de ejemplo tras el entrenamiento inicial

Los modelos, checkpoints y datasets generados no se suben al repositorio por motivos de tamaño y reproducibilidad.

## Nota sobre Hugging Face

Durante la descarga del dataset o del modelo puede aparecer un aviso relacionado con `HF_TOKEN`. Este aviso no bloquea la ejecución si el recurso es público, pero para ciertos modelos puede ser necesario iniciar sesión en Hugging Face o aceptar las condiciones de uso del modelo.

## Relación con el proyecto

Este notebook corresponde a la primera fase experimental del TFG. Su resultado sirve como punto de partida para las fases posteriores del pipeline:

1. generación de textos,
2. filtrado estadístico,
3. reentrenamiento iterativo,
4. evaluación de detectabilidad automática.
