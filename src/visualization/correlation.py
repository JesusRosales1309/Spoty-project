import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
archivo = "C:\\Users\\JESUS\\Desktop\\Spoty Project\\data\\processed\\StreamsFeatures.xlsx"
df = pd.read_excel(archivo, sheet_name='Hoja1',usecols=['danceability','energy','loudness','speechiness','liveness','valence','tempo','acousticness','duration_ms'])
#print(df)
df=pd.DataFrame(df)
upp_mat = np.triu(df.corr())
fig=sns.heatmap(df.corr(), vmin = -1, vmax = +1, annot = True, cmap = 'coolwarm', mask = upp_mat)
plt.show()
