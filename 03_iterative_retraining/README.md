# 03 - Reentrenamiento iterativo

Este directorio contiene el notebook encargado de continuar el ajuste fino del modelo a partir de los textos seleccionados en la iteración anterior.

Esta fase forma parte del ciclo iterativo del proyecto. Después de generar textos y seleccionar aquellos con mayor cercanía estadística al comportamiento humano, se construye un nuevo dataset mixto y se reentrena el adaptador LoRA correspondiente.

## Archivo incluido

* `03_iterative_retraining.py`: notebook de reentrenamiento iterativo del modelo.

## Objetivo

El objetivo de este notebook es continuar el entrenamiento del modelo generativo utilizando una combinación de textos humanos y textos generados seleccionados.

De esta forma, el modelo se va ajustando progresivamente en cada iteración, permitiendo analizar cómo evolucionan sus métricas estadísticas y su detectabilidad automática.

## Flujo general

1. Carga de los textos seleccionados en la iteración anterior.
2. Carga del conjunto humano de referencia.
3. Construcción de un dataset mixto.
4. Carga del modelo base `mistralai/Mistral-7B-v0.1`.
5. Carga del adaptador LoRA de la iteración anterior.
6. Tokenización del dataset mixto.
7. División en entrenamiento y evaluación.
8. Continuación del fine-tuning.
9. Evaluación del modelo reentrenado.
10. Guardado del nuevo adaptador LoRA.
11. Generación de un ejemplo de texto con la nueva iteración.

## Dataset mixto

El notebook combina dos tipos de textos:

* textos generados seleccionados en la iteración anterior,
* textos humanos procedentes del conjunto de referencia inicial.

El objetivo de esta mezcla es evitar que el modelo se reentrene únicamente sobre texto generado. Al mantener una parte de textos humanos, el proceso conserva una referencia lingüística real durante las sucesivas iteraciones.

En el código, esta mezcla se realiza mediante:

```python
mixed = concatenate_datasets([selected_ds, human_sample]).shuffle(seed=42)
```

## Continuación del ajuste fino

A diferencia del primer notebook, en esta fase no se entrena un adaptador LoRA desde cero. En su lugar, se carga el adaptador correspondiente a la iteración anterior y se continúa su entrenamiento.

Ejemplo:

```python
model = PeftModel.from_pretrained(
    base_model,
    PREVIOUS_LORA_PATH,
    is_trainable=True
)
```

El parámetro `is_trainable=True` permite seguir actualizando los pesos del adaptador LoRA.

## Configuración de entrenamiento

El entrenamiento se realiza con una configuración conservadora:

* una única época,
* tasa de aprendizaje baja,
* batch reducido,
* acumulación de gradiente,
* optimizador en 8 bits,
* cuantización del modelo base en 4 bits.

Esta configuración busca continuar el ajuste del modelo sin provocar cambios demasiado bruscos ni un sobreajuste excesivo a los textos seleccionados.

## Relación con las iteraciones

Cada ejecución de este notebook toma como entrada los textos seleccionados en una iteración y genera un nuevo adaptador LoRA para la siguiente.

Por ejemplo:

```text
selected_iter4_for_iter5.jsonl  ->  iter5_academic
```

Esto significa que los textos seleccionados tras la iteración 4 se utilizan para entrenar el adaptador correspondiente a la iteración 5.

## Salidas generadas

El notebook genera:

* un dataset mixto en formato `.jsonl`,
* métricas de evaluación del entrenamiento,
* un nuevo adaptador LoRA,
* un ejemplo de texto generado con la nueva iteración.

Los modelos, adaptadores y datasets generados no se incluyen en el repositorio debido a su tamaño.

## Relación con el pipeline completo

Este notebook representa la tercera fase del pipeline experimental:

1. fine-tuning inicial del modelo,
2. generación y selección de textos,
3. reentrenamiento iterativo,
4. nueva generación de textos,
5. repetición del ciclo.

Gracias a esta estructura, el proyecto permite estudiar cómo evolucionan los textos generados tras sucesivos ciclos de generación, selección y ajuste fino.

## Nota de reproducibilidad

Para reproducir una iteración concreta es necesario disponer de:

* el adaptador LoRA de la iteración anterior,
* el archivo `.jsonl` con los textos seleccionados,
* el conjunto humano de referencia,
* acceso al modelo base usado en el experimento.

Estos archivos no se incluyen directamente en el repositorio por restricciones de tamaño, pero el notebook mantiene explícitas las rutas y nombres utilizados durante la experimentación.
