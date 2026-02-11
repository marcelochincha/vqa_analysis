import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Crear datos aleatorios (Matriz de Similitud 12x12)
# Generamos valores entre 0 y 1
np.random.seed(42)
data = np.random.random((12,12)) * 0.4
# Forzamos un poco de "estructura" para que se vea bonito
# Hacemos que los primeros 4 (Lima) se parezcan entre ellos
data[0:4, 0:4] += 0.6
data[4:8, 4:8] += 0.6
data[8:12, 8:12] += 0.6
 

# 2. Definir los grupos y sus colores (Los tuyos)
grupos = ['Lima']*4 + ['NYC']*4 + ['VLM']*4
colores_dict = {'Lima': '#D55E00', 'NYC': '#56B4E9', 'VLM': '#009E73'}
colores_agentes = [colores_dict[g] for g in grupos]

# 3. Crear el DataFrame
df = pd.DataFrame(data, 
                  index=grupos,
                  columns=grupos)

# 4. El Clustermap
g = sns.clustermap(
    df,
    row_colors=colores_agentes, # Barra de color a la izquierda
    col_colors=colores_agentes, # Barra de color arriba
    cmap="magma",               # Escala de calor (negro-rojo-amarillo)
    linewidths=.5,
    figsize=(8, 8)
)

plt.show()