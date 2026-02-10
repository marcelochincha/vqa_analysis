# VQA Analysis Suite

Suite modular de análisis y visualización para comparar respuestas de VLMs y humanos en tareas de VQA.

## Estructura

```
src/
├── config.py              # Configuración: paths, parámetros de plots
├── utils.py               # Funciones compartidas (load, save, colores)
├── heatmap_common.py      # Utilidades de plotting para heatmaps
├── heatmap_bert.py        # Análisis BERT Score con auto-cálculo
├── heatmap_smatch.py      # Análisis SMATCH con parsing AMR
├── heatmap_stsb.py        # Análisis STSB-RoBERTa con auto-cálculo
├── embed_analysis.py      # Visualización embeddings (UMAP/PCA)
├── bias_analysis.py       # Análisis de bias por región
└── run_all.py             # Ejecutar todos los análisis
```

## Uso

### Ejecutar desde la raíz del workspace

**Todos los scripts deben ejecutarse desde la raíz del workspace** (`F:\robusto\vqa_analysis`), no desde `src/`.

### Scripts individuales

Cada análisis puede ejecutarse de forma independiente:

```powershell
# Embedding analysis
python src/embed_analysis.py

# Bias analysis
python src/bias_analysis.py

# BERT heatmaps
python src/heatmap_bert.py

# SMATCH heatmaps (incluye parsing AMR)
python src/heatmap_smatch.py

# STSB-RoBERTa heatmaps
python src/heatmap_stsb.py
```

### Ejecutar todo

```powershell
python src/run_all.py
```

## Métricas

### BERT Score
- **Rango**: 0-1 (F1 score)
- **Método**: Token-level semantic similarity
- **Salida**: `outputs/output_heatmap/output_scores/` y `output_plots/`
- **Auto-cálculo**: Si no existen los CSVs, los calcula automáticamente

### SMATCH
- **Rango**: 0-1 (F1 score)
- **Método**: AMR graph structural matching
- **Paso extra**: Parsea respuestas a AMR graphs con amrlib
- **Salida**: `outputs/output_smatch/output_scores/` y `output_plots/`
- **Auto-cálculo**: Parsing + scoring si no existen CSVs

### STSB-RoBERTa
- **Rango**: 0-1 (normalizado desde 0-5)
- **Método**: Cross-encoder semantic similarity
- **Salida**: `outputs/output_stsb_roberta/output_scores/` y `output_plots/`
- **Auto-cálculo**: Si no existen los CSVs, los calcula automáticamente

## Análisis

### Embedding Analysis
- **Genera embeddings** de todas las respuestas textuales
- **Reducción dimensional**: UMAP y PCA (2D)
- **Outputs**: 
  - `outputs/output_embeddings/umap_embeddings.png`
  - `outputs/output_embeddings/pca_embeddings_2d.png`

### Bias Analysis
- **Extrae ratings** numéricos de respuestas
- **Compara VLMs vs humanos** por región (Lima, NYC, All)
- **Outputs**:
  - `outputs/output_bias/bias_heatmap_{region}.png`
  - `outputs/output_bias/bias_distribution_{region}.png`

### Heatmap Analysis
- **Similarity heatmaps**: Matriz agente × agente
- **Agreement heatmaps**: Categorizado por nivel de acuerdo
- **Por bloques**: 4 bloques de 5 preguntas cada uno
- **Outputs**: 8 plots por métrica (4 bloques × 2 tipos)

## Configuración

Edita [config.py](config.py) para ajustar:
- **Paths**: Ubicación de inputs/outputs
- **Plot params**: Figsize, DPI, colormaps
- **Agent groups**: Agentes por región/tipo
- **Colors**: Esquema de colores (Lima=azul, NYC=verde, VLM=rojo)

## Dependencias

Instalar con:
```powershell
pip install -r requirements.txt
```

Para SMATCH, además:
```powershell
python -m amrlib.download model_parse_xfm_bart_large
```

## Notas

- **Modular**: Cada métrica tiene su propio script
- **Auto-detección**: Calcula métricas si no existen CSVs
- **Lógica custom**: Fácil modificar parámetros específicos por métrica
- **Sin clases**: Solo funciones, más simple de testear y modificar
