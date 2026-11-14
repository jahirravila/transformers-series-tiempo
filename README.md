# Transformers — Mecanismo de atención y Transformer básico para series de tiempo



Implementación **desde cero en TensorFlow + Keras** de un mecanismo de atención y de un Transformer (Vaswani et al., 2017) para pronosticar la **temperatura mínima diaria de los próximos 7 días en Melbourne** a partir de los 60 días anteriores.

| # | Notebook | Contenido |
|---|---|---|
| 01 | `01_datos_preprocesamiento.ipynb` | Análisis exploratorio, partición cronológica, escalado, codificación cíclica del calendario, ventanas 60 → 7 y modelos de referencia |
| 02 | `02_mecanismo_atencion.ipynb` | Atención de producto escalado en NumPy (pronóstico por analogía), LSTM con atención aditiva y de producto escalado, mapas de atención |
| 03 | `03_transformer_basico.ipynb` | Codificación posicional, multi-head attention propia (verificada contra Keras), Transformer-Encoder, ablación sin codificación posicional y Transformer Encoder-Decoder con máscara causal |

## Resultados (test 1989–1990, MAE en °C)

| Modelo | MAE 7 días | MAE día 1 | Parámetros |
|---|---|---|---|
| **Transformer Encoder-Decoder** | **1.940** | **1.703** | 30 209 |
| LSTM + atención aditiva | 1.981 | 1.866 | 22 471 |
| Transformer-Encoder | 1.983 | 1.883 | 21 831 |
| LSTM | 1.987 | 1.822 | 17 863 |
| Atención no paramétrica (NumPy) | 1.990 | 1.862 | 0 |
| Regresión lineal (Ridge) | 1.994 | 1.765 | — |
| Climatología | 2.024 | 2.031 | — |
| Persistencia | 2.624 | 1.958 | — |

## Ejecución en Google Colab
Abra los notebooks en orden (01 → 02 → 03) y use *Entorno de ejecución → Ejecutar todas*. El dataset se descarga automáticamente desde GitHub; todo corre en CPU (≈ 5 min) con semilla 42.

## Estructura
```
notebooks/   01, 02, 03 (ejecutados) · data/ · figures/ · results/ · modelos/
src/         ts_utils.py (datos y métricas) · attention_layers.py (atención, codificación posicional, MHA, bloques encoder/decoder)
docs/        informe técnico
```

## Referencias principales
Vaswani et al. (2017), *Attention Is All You Need* · Bahdanau et al. (2015) · Towards Data Science (2021), *Transformers Explained Visually*.
