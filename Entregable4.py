import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv(r"C:\Users\ASUS\Downloads\Bootcamp\Bases de datos-pandas\Bases de datos\ejercicios clase\steam-200k.csv")

#Indice 
# 20 - Exploración de los datos
# 218 - Preparación de los datos
# 


#######  1. Exploración de los datos ########

#print(df.info())
# #   Column                      Non-Null Count   Dtype  
#---  ------                      --------------   -----  
# 0   151603712                   199999 non-null  int64  
# 1   The Elder Scrolls V Skyrim  199999 non-null  str    
# 2   purchase                    199999 non-null  str    
# 3   1.0                         199999 non-null  float64
# 4   0                           199999 non-null  int64 
#print(df.describe())
#          151603712            1.0         0
#count  1.999990e+05  199999.000000  199999.0
#mean   1.036556e+08      17.874468       0.0
#std    7.208084e+07     138.057292       0.0
#min    5.250000e+03       0.100000       0.0
#25%    4.738420e+07       1.000000       0.0
#50%    8.691201e+07       1.000000       0.0
#75%    1.542309e+08       1.300000       0.0
#max    3.099031e+08   11754.000000       0.0
#print(df.shape) # (199999, 5)
#print(df.columns) 
# ['151603712', 'The Elder Scrolls V Skyrim', 'purchase', '1.0', '0']
#print(df.isnull().sum())
#151603712                     0
#The Elder Scrolls V Skyrim    0
#purchase                      0
#1.0                           0
#0                             0
#print(df.duplicated().sum()) #707

#  Eliminar la columna "0" ya que no tiene informacion

df = df.drop(columns=["0"])

#--------------------dar nombre a las columnas -----------------
df.columns = [
      "user-id",
      "game-title",
      "behavior-name",
      "value"
  ]

#print(df.head())
#     user-id                  game-title behavior-name  value
#0  151603712  The Elder Scrolls V Skyrim          play  273.0
#1  151603712                   Fallout 4      purchase    1.0
#2  151603712                   Fallout 4          play   87.0
#3  151603712                       Spore      purchase    1.0
#4  151603712                       Spore          play   14.9

###  Comportamientos
#print(df["behavior-name"].value_counts())
#behavior-name
#purchase    129510
#play         70489

### cantidad de usuarios
#print(df["user-id"].nunique()) #12393

### cantidad de videojuegos
#print(df["game-title"].nunique())  #5155

### Videojuegos más frecuentes
#print(df["game-title"].value_counts().head(10)) 
#game-title
#Dota 2                             9682
#Team Fortress 2                    4646
#Counter-Strike Global Offensive    2789
#Unturned                           2632
#Left 4 Dead 2                      1752
#Counter-Strike Source              1693
#Counter-Strike                     1424
#Garry's Mod                        1397
#The Elder Scrolls V Skyrim         1393
#Warframe                           1271

###Actividad por usuario
actividad_usuario = df.groupby("user-id").size()
#print(actividad_usuario.describe())

#count    12393.000000
#mean        16.138062
#std         51.878442
#min          1.000000
#25%          2.000000
#50%          3.000000
#75%          9.000000
#max       1573.000000

###  Análisis de comportamientos

porcentajes = df["behavior-name"].value_counts(normalize=True) * 100
#print(porcentajes)

#behavior-name
#purchase    64.755324
#play        35.244676

################################   Visualizacion  ################################

# Filtrar compras
df_purchase = df[df["behavior-name"] == "purchase"]

# Cantidad de compras por usuario
compras_usuario = df_purchase.groupby("user-id").size()

# Ordenar de mayor a menor
compras_usuario = compras_usuario.sort_values(ascending=False)

# Graficar
#plt.figure(figsize=(12,5))

#compras_usuario.head(20).plot(kind="bar")

#plt.title("Top 20 usuarios con más compras")
#plt.xlabel("ID del usuario")
#plt.ylabel("Cantidad de compras")

#plt.xticks(rotation=45)
#plt.grid(axis="y", alpha=0.7)
#plt.show()

# el jugador que mas compra ha realizado mas de 1000 compras los que siguen han realizado casi 800 compras 
# los siguientes jugadores hasta llegar a los 20 mas compradores estan entre 600 y 400 compras 

#------------------------------------------------------------------------------

# Filtrar videojuegos jugados
df_play = df[df["behavior-name"] == "play"]

# Cantidad de videojuegos diferentes por usuario
juegos_usuario = df_play.groupby("user-id")["game-title"].nunique()

# Ordenar
juegos_usuario = juegos_usuario.sort_values(ascending=False)

#Graficar
#plt.figure(figsize=(12,5))

#juegos_usuario.head(20).plot(kind="bar")

#plt.title("Top 20 usuarios con más videojuegos")
#plt.xlabel("ID del usuario")
#plt.ylabel("Cantidad de videojuegos")

#plt.xticks(rotation=45)
#plt.grid(axis="y", alpha=0.7)
#plt.show()

#el jugador con mas videojuegos tiene casi 500, los siguientes estan entre 300 y 200 videojuegos hasta completar los 20 que tienen mas juegos

#------------------------------------------------------------------------------

# Tiempo total de juego por usuario
tiempo_usuario = df_play.groupby("user-id")["value"].sum()

# Ordenar
tiempo_usuario = tiempo_usuario.sort_values(ascending=False)

#plt.figure(figsize=(12,5))

#tiempo_usuario.head(20).plot(kind="bar")

#plt.title("Top 20 usuarios con mayor tiempo de juego")
#plt.xlabel("ID del usuario")
#plt.ylabel("Tiempo de juego")

#plt.xticks(rotation=45)
#plt.grid(axis="y", alpha=0.7)
#plt.show()

# Los pimeros 4 jugadores invierten entre 10000 y 12000 horas los siguientes 6 entre 8000 y 10000 horas 
# los demas entre 7000 y 8000 horas

#------------------------------------------------------------------------------

# Crear tabla de comportamiento por usuario
comportamiento_usuario = pd.DataFrame({
    "compras": compras_usuario,
    "juegos_diferentes": juegos_usuario,
    "tiempo_juego": tiempo_usuario
})

# Reemplazar valores faltantes por cero
comportamiento_usuario = comportamiento_usuario.fillna(0)

#print(comportamiento_usuario.head())

#         compras  juegos_diferentes  tiempo_juego
#user-id                                          
#5250          21                6.0         225.5
#76767         36               20.0        1227.0
#86540         82               15.0         255.0
#103360        10                0.0           0.0
#144736         8                1.0           0.1

#######  2. Preparación de los datos ########

# Cantidad de duplicados
#print("Duplicados:", df.duplicated().sum())
# Duplicados: 707

# Duplicados por comportamiento
#print(df[df.duplicated(keep=False)]["behavior-name"].value_counts())
#behavior-name
#purchase    1414

# Separar comportamientos
df_purchase = df[df["behavior-name"] == "purchase"]
df_play = df[df["behavior-name"] == "play"]

# Compras por usuario
compras_usuario = df_purchase.groupby("user-id").size()

# Juegos diferentes por usuario
juegos_usuario = df_play.groupby("user-id")["game-title"].nunique()

# Tiempo total de juego por usuario
tiempo_usuario = df_play.groupby("user-id")["value"].sum()

# Construir tabla
comportamiento_usuario = pd.DataFrame({
    "compras": compras_usuario,
    "juegos_diferentes": juegos_usuario,
    "tiempo_juego": tiempo_usuario
}).fillna(0)

# Convertir cantidades a enteros
comportamiento_usuario["compras"] = comportamiento_usuario["compras"].astype(int)
comportamiento_usuario["juegos_diferentes"] = comportamiento_usuario["juegos_diferentes"].astype(int)

#print(comportamiento_usuario.head())
#         compras  juegos_diferentes  tiempo_juego
#user-id                                          
#5250          21                  6         225.5
#76767         36                 20        1227.0
#86540         82                 15         255.0
#103360        10                  0           0.0
#144736         8                  1           0.1


#print((comportamiento_usuario["compras"] == 0).sum()) #0
#print((comportamiento_usuario["juegos_diferentes"] == 0).sum()) #1043
#print((comportamiento_usuario["tiempo_juego"] == 0).sum()) #1043

# ------------ Seleccionar las variables para clustering ------------

X = comportamiento_usuario[
    ["compras", "juegos_diferentes", "tiempo_juego"]
].copy()

#print(X.head())
#         compras  juegos_diferentes  tiempo_juego
#user-id                                          
#5250          21                  6         225.5
#76767         36                 20        1227.0
#86540         82                 15         255.0
#103360        10                  0           0.0
#144736         8                  1           0.1
#print(X.shape)
#(12393, 3)

#----------------Escalador ---------------------

# Crear escalador
scaler = StandardScaler()

# Ajustar y transformar
X_scaled = scaler.fit_transform(X)

# Convertir a DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns,
    index=X.index
)

#print(X_scaled.head())
#          compras  juegos_diferentes  tiempo_juego
#user-id                                           
#5250     0.294170           0.018337     -0.069069
#76767    0.712431           0.838113      1.248278
#86540    1.995099           0.545336     -0.030265
#103360  -0.012555          -0.332995     -0.365686
#144736  -0.068323          -0.274440     -0.365554

# Guardar resultados
inercias = []
siluetas = []

# Evaluar entre 2 y 10 clusters
rangos = range(2, 11)

for k in rangos:

    modelo = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo.fit_predict(X_scaled)

    # Método del codo
    inercias.append(modelo.inertia_)

    # Silhouette
    silueta = silhouette_score(X_scaled,etiquetas,sample_size=3000,random_state=42)

    siluetas.append(silueta)

# ------------------ Grafica del metodo del codo ------------------
#plt.figure(figsize=(8,5))

#plt.plot(
#    list(rangos),
#    inercias,
#    marker="*"
#)

#plt.title("Método del codo")
#plt.xlabel("Cantidad de clusters (K)")
#plt.ylabel("Inercia")
#plt.xticks(list(rangos))
#plt.grid(alpha=0.5)

#plt.show()

#segun la grafica se identifica que se empieza a estabilizar en 5

# ------------------ Grafica del metodo Silhouette ------------------

#plt.figure(figsize=(8,5))

#plt.plot(
#    list(rangos),
#    siluetas,
#    marker="o"
#)

#plt.title("Coeficiente Silhouette")
#plt.xlabel("Cantidad de clusters (K)")
#plt.ylabel("Puntuación Silhouette")
#plt.xticks(list(rangos))
#plt.grid(alpha=0.5)

#plt.show()

# se identifica que desde el # 5 se empieza a disminuir de forma cnotable la diferencia
# entre cada uno de los grupos

#-------------------- Elegir entre PCA y t-SNE --------------------
# debido a que hay variable numericas, es mas sencillo de interpretar y a que permite identificar la proporcion de cantidad
# de informacion que explica cada componente se elige PCA y debido a que el analisis que se busca es lineal

# debido a que tiene pocas variables y el analisis no es complejo no se elige T-SNE

# Crear modelo PCA
pca = PCA(n_components=2)

# Ajustar y transformar
X_pca = pca.fit_transform(X_scaled)

# Crear DataFrame
df_pca = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=X_scaled.index
)

#print(df_pca.head())
#              PC1       PC2
#user-id                    
#5250     0.158136 -0.184006
#76767    1.562842  0.570396
#86540    1.547804 -0.981801
#103360  -0.392838 -0.222115
#144736  -0.390097 -0.215651

#######  3. Modelado ########

## ------------- Clusters -------------------

# Definir cantidad de clusters
k_optimo = 5

# Crear modelo
kmeans = KMeans(
    n_clusters=k_optimo,
    random_state=42,
    n_init=10
)

# Entrenar modelo
clusters = kmeans.fit_predict(X_scaled)

# Asignar resultados
comportamiento_usuario["cluster"] = clusters

#print(comportamiento_usuario.head())
#         compras  juegos_diferentes  tiempo_juego  cluster
#user-id                                                   
#5250          21                  6         225.5        0
#76767         36                 20        1227.0        4
#86540         82                 15         255.0        4
#103360        10                  0           0.0        0
#144736         8                  1           0.1        0

### ---------- Grafica de visualizacion de grupo por PCA -----------

#plt.figure(figsize=(10,7))

#sns.scatterplot( 
#    x=df_pca["PC1"],
#    y=df_pca["PC2"],
#    hue=comportamiento_usuario["cluster"],
#    palette="viridis",
#    alpha=0.6,
#    s=35,
#    hue_order=[0, 1, 2, 3, 4]
#)

#plt.title("Segmentación de usuarios mediante K-Means y PCA")
#plt.xlabel("Componente principal 1")
#plt.ylabel("Componente principal 2")

#plt.legend(title="Clusters")
#plt.grid(alpha=0.3)

#plt.show()

#cantidad de usuarios por Cluster
cantidad_clusters = comportamiento_usuario["cluster"].value_counts().sort_index()

#print(cantidad_clusters)
#cluster
#0    10822
#1      124
#2       44
#3      244
#4     1159

#plt.figure(figsize=(8,5))

#sns.countplot(
#    data=comportamiento_usuario,
#    x="cluster",
#    hue="cluster",
#    palette="viridis",
#    legend=False
#)

#plt.title("Cantidad de usuarios por cluster")
#plt.xlabel("Cluster")
#plt.ylabel("Cantidad de usuarios")

#plt.grid(axis="y", alpha=0.3)
#plt.show()

#promedio de cada uno de los clusters
perfil_clusters = comportamiento_usuario.groupby("cluster")[
    ["compras", "juegos_diferentes", "tiempo_juego"]
].mean()

#print(perfil_clusters)
#            compras  juegos_diferentes  tiempo_juego
#cluster                                             
#0          3.571798           2.048790     72.178156
#1         31.564516          20.129032   4940.769355
#2        435.318182         184.568182   3508.886364
#3        138.127049          74.934426   2006.608197
#4         29.408973          16.742019   1214.490940

perfil_clusters.plot(
    kind="bar",
    figsize=(11,6)
)

#plt.title("Comportamiento promedio de los usuarios por cluster")
#plt.xlabel("Cluster")
#plt.ylabel("Promedio de actividad")

#plt.xticks(rotation=0)
#plt.legend(title="Variables")

#plt.grid(axis="y", alpha=0.3)
#plt.show()

silhouette_final = silhouette_score(
    X_scaled,
    comportamiento_usuario["cluster"],
    sample_size=3000,
    random_state=42
)

#print("Silhouette del modelo:", silhouette_final)
#Silhouette del modelo: 0.7733802252974167

#######  4. Interpretación de los clusters ########

#Cluster 0  - Jugadores ocasionales
#corresponde al 87.32% de los usuarios
#10822 usuarios
#compran en promedio 3.57
#tienen 2.05 juegos 
#invierten 72.18 horas en juegos_usuario

#Es el grupo que tiene la mayor cantidad de usuarios, 
#sin emabrgo tienen pocas compras, juegan poco con los 
#videojuegos y tienen poco tiempo de juego. 

#Cluster 1 - Jugadores intensivos
#Representado por el 1% de los jugadores
#Con 124 usuarios
#tienen un promerio de compras de 31.56
#tienen 20.13 juegos 
#e invierten 4940.77 horas juegos_usuario

#Es un grupo que tiene una cantidad moderada de compras y de 
#videojuegos diferentes y son los qu einvierten mas tiempo en jugar

#Cluster 2 - Coleccionistas de Videojuegos
#cuenta con el 0.35% de los usuarios
#con 44 usuarios
#435.32 compras en promedio 
#184.57 videojuegos diferentes
#e invierten 3508.89 horas

#Por lo anterior es posible identificar que son los que tienen 
#mas compras y videojuegos, son los que tienen mas juegos. 
#y tienen una considerable cantidad de horas de juego invertidas

#Cluster 3 - Usuarios Activos 
#corresponde al 1.97% de los usuarios
#con 244 usuarios
#138.13 compras en promedio
#74.93 videojuegos
#e invierten 2006.61 horas

#Compran bastantes juegos, aunque juegan mas juegos, y tienen muchas 
#horas de juego, son los uaurios que son mas equilibrados en de todos 
#los grupos.

#Cluster 4 - Jugadores frecuentes
#corresponde al 9.35% de los usuarios
#con 1159 usuarios
#con compras de 29.41
#tiene 16.74 videojuegos
#e invierte 1214.49 horas de juegos_usuario

#por tanto son los jugadores mas equilibrados tanto en 
#compras, juegos y timepo invertido aunque esto no los hace especializados

#----------------------Visualizacion de los perfiles ----------------------

# Normalizar promedios de los clusters
scaler_radar = MinMaxScaler()

perfil_normalizado = pd.DataFrame(
    scaler_radar.fit_transform(perfil_clusters),
    columns=perfil_clusters.columns,
    index=perfil_clusters.index
)

# Gráfica
categorias = perfil_normalizado.columns.tolist()

valores = perfil_normalizado.values

angulos = np.linspace(
    0, 2 * np.pi,
    len(categorias),
    endpoint=False
).tolist()

angulos += angulos[:1]

plt.figure(figsize=(9,9))

ax = plt.subplot(111, polar=True)

for i, cluster in enumerate(perfil_normalizado.index):

    datos = valores[i].tolist()
    datos += datos[:1]

    ax.plot(
        angulos,
        datos,
        label=f"Cluster {cluster}"
    )

    ax.fill(angulos, datos, alpha=0.08)

ax.set_xticks(angulos[:-1])
ax.set_xticklabels(categorias)

plt.title("Comparación de perfiles de usuarios")
plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.show()


#######  5. Recomendaciones de negocio ########

#cluster 0 : Jugadores ocasionales
#Implementar campañas de activación mediante recomendaciones personalizadas,
#novedades y contenido que motive a descubrir nuevos videojuegos.

#descuentos de bienvenida, promociones para probar nuevos videojuegos y 
# recomendaciones de títulos gratuitos o económicos. El objetivo sería 
# incentivar su participación y aumentar su actividad.

#campañas de bajo costo para aumentar las compras

#cluster 1 : Jugadores intensivos
#Crear campañas de engagement, promocionar actualizaciones, expansiones y 
#contenido adicional de los juegos que utilizan frecuentemente.

#descuentos en expansiones, DLC, contenido adicional y productos 
# relacionados con sus videojuegos favoritos. También podrían recibir 
# beneficios por mantenerse activos.

#también sería importante debido a su alto tiempo de juego. Se podrían 
# desarrollar campañas para mantener su participación y motivar la 
# adquisición de contenido adicional.

#cluster 2 : Coleccionistas de videojuegos
#Ofrecer recomendaciones de nuevos lanzamientos, colecciones, paquetes de
#videojuegos y acceso anticipado a novedades.

#promociones por compras múltiples, paquetes de videojuegos, descuentos 
# por colecciones y ofertas especiales para nuevos lanzamientos.

#merece atención por su elevado número de compras y videojuegos diferentes.
#  Se podrían implementar estrategias para conservar su interés y motivar 
# futuras adquisiciones.

#cluster 3 : Usuarios altamente activos
#Desarrollar estrategias de fidelización VIP, beneficios exclusivos y 
#programas de recompensas para fortalecer su relación con la plataforma.

#recompensas exclusivas, acceso anticipado, beneficios VIP y descuentos 
# especiales por fidelidad. Se buscaría reconocer su alto nivel de 
# participación.

#sería una prioridad porque presenta un comportamiento de alta actividad 
# en las tres variables. Es importante mantener su satisfacción y 
# fortalecer su relación con la plataforma.

#cluster 4 : Jugadores frecuentes
#Incentivar nuevas compras mediante recomendaciones relacionadas con 
#sus preferencias y promociones de categorías similares.

#descuentos personalizados, promociones por categorías de videojuegos 
#de su interés y ofertas complementarias para incentivar nuevas compras.

#campañas de retencion y crecimiento