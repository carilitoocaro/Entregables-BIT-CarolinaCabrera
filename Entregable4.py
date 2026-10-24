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


df = pd.read_csv(r"C:\Users\ASUS\Downloads\Bootcamp\Bases de datos-pandas\Bases de datos\ejercicios clase\steam-200k.csv")

#######  1. Exploración de los datos ########


#print(df.head())
#   151603712  The Elder Scrolls V Skyrim  purchase    1.0  0
#0  151603712  The Elder Scrolls V Skyrim      play  273.0  0
#1  151603712                   Fallout 4  purchase    1.0  0
#2  151603712                   Fallout 4      play   87.0  0
#3  151603712                       Spore  purchase    1.0  0
#4  151603712                       Spore      play   14.9  0

#df.info()
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

#----------------------------------------------------------------
#En la descripcion de la BD en Kaggle se indica que  
# - "This dataset Is a list of user behaviors, with columns: 
# user-id, game-title, behavior-name, value" por lo tanto de 
# elimina la fila 5 ya que solo contiene (0)
#----------------------------------------------------------------

df = df.drop(columns=["0"])
#print(df.head())
#   151603712  The Elder Scrolls V Skyrim  purchase    1.0
#0  151603712  The Elder Scrolls V Skyrim      play  273.0
#1  151603712                   Fallout 4  purchase    1.0
#2  151603712                   Fallout 4      play   87.0
#3  151603712                       Spore  purchase    1.0
#4  151603712                       Spore      play   14.9

df.columns = ["user_id", "game_title", "behavior_name", "value"]
#print(df.info())
# #   Column         Non-Null Count   Dtype  
#---  ------         --------------   -----  
# 0   user_id        199999 non-null  int64  
# 1   game_title     199999 non-null  str    
# 2   behavior_name  199999 non-null  str    
# 3   value          199999 non-null  float64

#----------------------------------------------------------------
#Con el fin de verificar que se haya eliminado la comlumna que 
# no tiene valores se ejecuta de nuevo .describe() y veriifcar 
# que no hayan alteraciones o modificaciones adicionales en la 
# BD:
#----------------------------------------------------------------

#print(df.describe())
#            user_id          value
#count  1.999990e+05  199999.000000
#mean   1.036556e+08      17.874468
#std    7.208084e+07     138.057292
#min    5.250000e+03       0.100000
#25%    4.738420e+07       1.000000
#50%    8.691201e+07       1.000000
#75%    1.542309e+08       1.300000
#max    3.099031e+08   11754.000000

#----------------------------------------------------------------
#En cuanto al User ID, este se puede repetir ya que el mismo 
# usuario puede tener diferentes comportamientos, por ello sus 
# duplicados no se eliminan y ayuda a tener una mejor idea del 
# comportamiento del usuario

#Por otra parte el valor representa segun la descripcion de BD 
# la cantidad de horas que un usuario ha jugado -  y es bastante
# claro el porque hay un valor como 1 ya que representa la 
# actividad de comprar.
#----------------------------------------------------------------

##Se identifica el tamaño de la BD

#print(df.shape)
#(199999, 4)

#Se identifica, si hay valores nulos
#print(df.isna().sum())
#user_id          0
#game_title       0
#behavior_name    0
#value            0

#----------------------------------------------------------------
#Se buscan los valores duplicados y se encuentran 707, ya que no 
# hay fechas o informacion adicional, se opta por dejar los 
# valores ya que pueden ser registros de dias diferentes y al 
# elimnarlos puede afectar el analisis de la BD
#----------------------------------------------------------------

#print(df.duplicated().sum())
#np.int64(707)

#----------------------------------------------------------------
#Se buscan los valores unicos y la cantidad de ellos por columna,
#  esto con el fin de ver si necesitan alguna modificacion. 

#Se encuentra que en "behavior_name", estan unificados y 
# correctos ['play', 'purchase'], que hay 12393 usuarios unicos 
# y que hay 5155 juegos unicos. 

#----------------------------------------------------------------
#print(df["behavior_name"].unique())
#['play', 'purchase']

#print(df["user_id"].nunique())
#12393

#print(df["game_title"].nunique())
#5155

#----------------------------------------------------------------
#Debido al alto numero de valores unicos con formato str en las 
# columnas correspondientes, se elige la opcion de unificar su 
# formato para que la informacion que se obtenga sea lo mas 
# uniforme posible, sin necesidad de revisar uno a uno los 
# valores de las columnas
#----------------------------------------------------------------

df["game_title"] = df["game_title"].str.lower().str.strip().str.title()
df["behavior_name"] = df["behavior_name"].str.lower().str.strip().str.title()

#######   Visualizacion  ####### 

#Con la informacion que proporciona la BD es porsible responder 
# la siguiente pregunta:

#¿Cuáles son los juegos a los que más tiempo dedican los usuarios?
#Para ello es necesario hacer un filtro de los juegos que tienen 
# como comportamiento "play"

#----------------------------------------------------------------
df_juego = df[df["behavior_name"] == 'Play']
#print(df_juego.head())
#     user_id                  game_title behavior_name  value
#0  151603712  The Elder Scrolls V Skyrim          Play  273.0
#2  151603712                   Fallout 4          Play   87.0
#4  151603712                       Spore          Play   14.9
#6  151603712           Fallout New Vegas          Play   12.1
#8  151603712               Left 4 Dead 2          Play    8.9

#Se verifica el tamaño de la BD
#print(df_juego.shape)
#(70489, 4)

#print(df_juego["behavior_name"].unique())
#['Play']

df_compra = df[df['behavior_name'] == "Purchase"]
#print(df_compra.head())
#     user_id         game_title behavior_name  value
#1  151603712          Fallout 4      Purchase    1.0
#3  151603712              Spore      Purchase    1.0
#5  151603712  Fallout New Vegas      Purchase    1.0
#7  151603712      Left 4 Dead 2      Purchase    1.0
#9  151603712           Huniepop      Purchase    1.0

#print(df_compra.shape)
#(129510, 4)

juego_tiempo = df_juego.groupby("game_title")["value"].sum().sort_values(ascending=False)
#print(juego_tiempo.head(10))
#game_title
#Dota 2                                         981684.6
#Counter-Strike Global Offensive                322771.6
#Team Fortress 2                                173673.3
#Counter-Strike                                 134261.1
#Sid Meier'S Civilization V                      99821.3
#Counter-Strike Source                           96075.5
#The Elder Scrolls V Skyrim                      70889.3
#Garry'S Mod                                     49725.3
#Call Of Duty Modern Warfare 2 - Multiplayer     42009.9
#Left 4 Dead 2                                   33596.7

juego10 = juego_tiempo.head(10)
#plt.figure(figsize=(10, 6))
#plt.barh(juego10.index, juego10.values)
#plt.xlabel('Tiempo (horas)')
#plt.ylabel('Juego')
#plt.title('Top 10 de juegos por tiempo jugado')
#plt.xticks(rotation=90)
#plt.grid(axis='x')
#plt.ticklabel_format(axis="x", style="plain", useOffset=False)
#plt.tight_layout()
#plt.show()

#----------------------------------------------------------------
#Es posible identificar que el juego que mas juegan los usuarios 
# es Dota 2 y le sigue Counter-Strike Global Offensive
#----------------------------------------------------------------

comportamiento = df["behavior_name"].value_counts()

#plt.figure(figsize=(10, 6))
#plt.pie(comportamiento.values, labels=comportamiento.index, autopct='%1.1f%%')
#plt.title('Distribución de comportamiento')
#plt.show()

#Se identifica que es importante conocer la cantidad de informacion 
# si se enfoca en comportamiento de compra o de juego, donde es posible 
# identificar que el 64% esta en compra y el exedente 35.2 en juego

#----------------------------------------------------------------

#Ahora podemos responder la siguiente pregunta 
#2. ¿Cuáles son los juegos que más se compran en Steam?

MasComprados = df_compra["game_title"].value_counts().head(10)
#print(MasComprados)
#game_title
#Dota 2                             4841
#Team Fortress 2                    2323
#Unturned                           1563
#Counter-Strike Global Offensive    1412
#Half-Life 2 Lost Coast              981
#Counter-Strike Source               978
#Left 4 Dead 2                       951
#Counter-Strike                      856
#Warframe                            847
#Half-Life 2 Deathmatch              823

MasComprados10 = MasComprados.head(10)
#plt.figure(figsize=(10, 6))
#plt.barh(MasComprados10.index, MasComprados10.values)
#plt.xlabel('Cantidad de compras')
#plt.ylabel('Juego')
#plt.title('Top 10 de juegos más comprados')
#plt.xticks(rotation=90)
#plt.grid(axis='x')
#plt.ticklabel_format(axis="x", style="plain", useOffset=False)
#plt.tight_layout()
#plt.show()

#Es posible identificar que el juego mas comprado es Dota 2 y le 
# sigue Team Fortress 2
#----------------------------------------------------------------

#Ahora tambien es posible responder la siguiente pregunta: 
# 3. ¿Cómo se distribuyen las horas de juego entre los usuarios?

horas_usuario = df_juego.groupby("user_id")["value"].sum().sort_values(ascending=False)
#print(horas_usuario.head(10))
#user_id
#73017395     11754.0
#10599862     11651.7
#100630947    10853.2
#26762388     10470.1
#153382649     9640.0
#43684632      9546.3
#48798067      9427.2
#52731290      9417.6
#42935819      8172.9
#14544587      8137.1

UsuariosHoras10 = horas_usuario.head(10)

#plt.figure(figsize=(10, 6))
#plt.bar(UsuariosHoras10.index.astype(str),UsuariosHoras10.values)

#plt.title("Top 10 usuarios con más horas de juego")
#plt.xlabel("ID del usuario")
#plt.ylabel("Horas acumuladas")

#plt.xticks(rotation=45)
#plt.grid(axis='y')
#plt.tight_layout()
#plt.show()

#Los usuarios que mas invierten tiempo estan entre 10.000 y 12.000 horas 
# jugando, que son los primeros 4

#----------------------------------------------------------------
#######  2. Preparación de los datos ########

#separar los registros - play (usuario jugó un juego) / purchase (usuario 
# compró un juego)

#sin emabrgo para esto ya esta df_juego y df_compra respectivamente
#----------------------------------------------------------------

juego_comprado = (df_compra.groupby("user_id")["game_title"].nunique())

df_grupo = pd.concat([horas_usuario, juego_comprado], axis=1)

df_grupo.colums = ["horas_jugadas", "juegos_comprados"] 

df_grupo = df_grupo.fillna(0)

#print("Dimensiones:", df_grupo.shape)
#print("Valores faltantes:")
#print(df_grupo.isna().sum())

#Dimensiones: (12393, 2)
#Valores faltantes:
#value         0
#game_title    0

#print(df_grupo.describe())
#              value    game_title
#count  12393.000000  12393.000000
#mean     278.009021     10.392964
#std      760.270618     35.635013
#min        0.000000      1.000000
#25%        1.300000      1.000000
#50%       13.500000      2.000000
#75%      155.600000      6.000000
#max    11754.000000   1068.000000

#print(df_grupo.head())
#user_id                       
#73017395   11754.0           1
#10599862   11651.7         257
#100630947  10853.2          10
#26762388   10470.1         276
#153382649   9640.0           2

#####  StandadScaler #####

#Escalado de los valores de la BD

scaler = StandardScaler()
Xscaled = scaler.fit_transform(df_grupo)

#print("Dimensiones de Xscaled:", Xscaled.shape)
#print("Primeras cinco filas:")
#print(Xscaled[:5])

#Dimensiones de Xscaled: (12393, 2)
#Primeras cinco filas:
#[[ 1.50952223e+01 -2.63598693e-01]
# [ 1.49606596e+01  6.92063699e+00]
# [ 1.39103333e+01 -1.10279076e-02]
# [ 1.34064134e+01  7.45384199e+00]
# [ 1.23145213e+01 -2.35535273e-01]]

#####  3. Modelado #####

####  Metodo de Codo #####

valores_k = range(2, 10)

inercia = []
siluetas = []

for cantidad in valores_k:

    modelo = KMeans(
        n_clusters=cantidad,
        random_state=42,
        n_init=10
    )

    etiquetas = modelo.fit_predict(Xscaled)

    inercia.append(modelo.inertia_)

    siluetas.append(
        silhouette_score(Xscaled, etiquetas)
    )

#Grafica para metodo de codo

#plt.figure(figsize=(8, 5))
#plt.plot(
#    list(valores_k),
#    inercia,
#    marker="o"
#)
#plt.xlabel("Número de clusters (k)")
#plt.ylabel("Inercia")
#plt.title("Método del codo - K-Means")
#plt.xticks(list(valores_k))
#plt.grid(alpha=0.3)
#plt.show()

####    K-Means  ####

#plt.figure(figsize=(8, 5))
#plt.plot(
#    list(valores_k),
#    siluetas,
#    marker="*"
#)
#plt.xlabel("Número de clusters (k)")
#plt.ylabel("Silhouette Score")
#plt.title("Evaluación de clusters - K-Means")
#plt.xticks(list(valores_k))
#plt.grid(alpha=0.3)
#plt.show()

#----------------------------------------------------------------
#En "k_elegido", se pone la cantidad de clusters (grupos que se desean 
# elegir) En este caso tanto en el metodo de codo y en silhouette se 
# empieza a estabilizar en 6

#donde es posible identificar que en el grupo 0 se encuentra la mayoria 
# de los datoscon 10.493 y en el grupo 1 y 3 esta la menor cantidad de 
# datos con 92 y 32 respectivamente
#----------------------------------------------------------------

#---- cantidad de datos por grupo de cluster -----
k_elegido = 6

kmeans = KMeans(
    n_clusters=k_elegido,
    random_state=42,
    n_init=10
)

df_grupo["cluster_kmeans"] = kmeans.fit_predict(
    Xscaled
)

#print(df_grupo["cluster_kmeans"].value_counts().sort_index())

#cluster_kmeans
#0    10493
#1       92
#2     1217
#3       32
#4      354
#5      205

#valores promedio de cada uno de los clusters

perfil_kmeans = df_grupo.groupby(
    "cluster_kmeans"
)[
    ["value", "game_title"]
].mean().round(2)

#print(perfil_kmeans)
         
#cluster_kmeans   value      game_title            
#0                 53.12        3.41
#1               5873.04       64.57
#2                849.57       24.76
#3               3112.34      494.06
#4               2437.68       27.03
#5               1713.04      154.24

#######  Segmentacion en Grafica con K-means  #######

#plt.figure(figsize=(10, 6))

#plt.scatter(
#    df_grupo["value"],
#    df_grupo["game_title"],
#    c=df_grupo["cluster_kmeans"],
#    cmap="viridis",
#    alpha=0.5
#)

#plt.xlabel("Horas jugadas")
#plt.ylabel("Juegos comprados")
#plt.title("Segmentación de usuarios con K-Means (K=6)")
#plt.grid(alpha=0.2)
#plt.tight_layout()
#plt.show()

######  DBSCAN ######

#Iniciar con 5 samples

min_samples_dbscan = 5

vecinos = NearestNeighbors(
    n_neighbors=min_samples_dbscan
)

vecinos.fit(Xscaled)

distancias, indices = vecinos.kneighbors(Xscaled)

#----------------------------------------------------------------
#Se genera la grafica con el fin de identificar la posicion de K, el 
# cual se podrá ver segun el codo de la grafica

#No es posible establecer el punto de inflexion en la grafica debido 
# a la cantidad de datos que esta muestra
#----------------------------------------------------------------

distancias_k = np.sort(
    distancias[:, -1]
)

#plt.figure(figsize=(10, 5))
#plt.plot(distancias_k)
#plt.xlabel("Usuarios ordenados por distancia")
#plt.ylabel("Distancia al vecino número 5")
#plt.title("Gráfica de distancias para ajustar DBSCAN")
#plt.grid(alpha=0.3)
#plt.show()

#----------------------------------------------------------------
#Se amplia la grafica en la parte del codo, para identificar el 
# punto en el que cambia y se identifica que se empieza a 
# normalizar en 5
#----------------------------------------------------------------

n = len(distancias_k)
inicio = int(n * 0.95)

#plt.figure(figsize=(10, 5))
#plt.plot(
#    range(inicio, n),
#    distancias_k[inicio:]
#)

#plt.xlabel("Posición de la observación")
#plt.ylabel("Distancia al quinto vecino")
#plt.title("Ampliación del codo de DBSCAN")
#plt.grid(alpha=0.3)
#plt.show()

eps_elegido = 0.4  # Ajustar según la gráfica

dbscan = DBSCAN(
    eps=eps_elegido,
    min_samples=min_samples_dbscan
)

etiquetas_dbscan = dbscan.fit_predict(Xscaled)

df_grupo["cluster_dbscan"] = etiquetas_dbscan

#print("Distribución de clusters:")
#print(df_grupo["cluster_dbscan"].value_counts().sort_index())
#print("Cantidad de grupos encontrados:",len(set(etiquetas_dbscan) - {-1}))
#print("Cantidad de usuarios clasificados como ruido:",(etiquetas_dbscan == -1).sum())

#----------------------------------------------------------------
#Distribución de clusters:
#cluster_dbscan
#-1      129
# 0    12230
# 1        6
# 2        5
# 3       16
# 4        7
#Name: count, dtype: int64
#Cantidad de grupos encontrados: 5
#Cantidad de usuarios clasificados como ruido: 129
#----------------------------------------------------------------

mascara = etiquetas_dbscan != -1
etiquetas_sin_ruido = etiquetas_dbscan[mascara]
X_sin_ruido = Xscaled[mascara]

n_clusters_dbscan = len(
    set(etiquetas_sin_ruido)
)

#if 2 <= n_clusters_dbscan < len(etiquetas_sin_ruido):

#    silhouette_dbscan = silhouette_score(
#        X_sin_ruido,
#        etiquetas_sin_ruido
#    )

#    print(
#        "Silhouette DBSCAN sin ruido:",
#        round(silhouette_dbscan, 4)
#    )

#else:
#    print(
#        "No se puede calcular Silhouette: "
#        "no hay suficientes grupos válidos."
#    )

#Silhouette DBSCAN sin ruido: 0.8521

#Grafica DBSCAN

#Luego de revisar la grafica los grupos que se identifican no se ven ni 
# diferencian con claridad, por tanto se ven los grupos sobrepuestos y 
# no divididos, por tanto se toma la decision de tomar el analisis de la 
# grafica de K-Means

#plt.figure(figsize=(10, 6))

#plt.scatter(
#    df_grupo["value"],
#    df_grupo["game_title"],
#    c=df_grupo["cluster_dbscan"],
#    cmap="plasma",
#    alpha=0.5
#)

#plt.xlabel("Horas jugadas")
#plt.ylabel("Juegos comprados")
#plt.title("Segmentación de usuarios con DBSCAN")
#plt.grid(alpha=0.2)
#plt.tight_layout()
#plt.show()

#######  Interpretación de los clusters  #######

# Se identifica que hay varios grupos dentro de los cuales se encuentran 
# grupos en los que:

#0 - "Los menos jugadores" es el grupo en el que menos tiempo se invierte 
# en juegos y el que menos juegos juegan el que tiene la mayor cantidad 
# de datos con 10.493 datos
#1 - "Jugadores compulsivos" los que mas compran juegos con 92 datos
#2 - "Jugadores y compradores moderados" con 1.217 datos
#3 - "compradores compulsivos" los que mas compran juegos con 32 datos
#4 - "Amantes de sus juegos" con 354 datos
#5 - "Encantados por comprar y jugar" con 205 datos

#cluster_kmeans --value ----game_title-------Cant. de datos
#0 ------------- 53.12-----3.41 -->Morado -------10493
#1 ------------- 5873.04 --64.57 -->Violeta ------92
#2 ------------- 849.57 ---24.76 -->Turquesa -----1217
#3 ------------- 3112.34 --494.06 --Aguamarina ----32
#4 ------------- 2437.68 --27.03 -->Verde --------354
#5 ------------- 1713.04 --154.24 -->Amarillo -----205

#1. Qué caracteriza a los usuarios de este grupo? 
# Se identifica que en la Base de Datos en los registros en general se 
# los jugadores en su mayoria invierten mas tiempo jugando que 
# comprando juegos

#2. Son jugadores ocasionales o frecuentes? 
# en los grupos 1, 3, 4 y 5 se encuentran los usuarios que invierten 
# mas tiempo jugando

#3. Presentan comportamientos similares de compra? 
# en cuanto a los comportamientos de compra, se identifica que no tienen 
# un alto nivel de compra

#4. Qué diferencias existen entre los clusters? 
# se identifica que el grupo en la mayor cantidad de tiempo invertida en 
# jugar esta en el grupo 1 y 3, los que menos compras tienen con el 0 y 
# el 2 y la mayor cantidad de datos se encuentran en los grupos 0 y 2


#######  Recomendaciones de negocio  #######

#1. Qué estrategia de marketing implementarías para cada grupo?
####0: para este grupo por comprar juegos y pasar niveles, les daria 
# puntos adicionales coln los que puedan tener creditos que les 
# permitan tener descuentos en la compra de los siguientes juegos

####1: por jugar tanto, crearia por temporadas campañas de compra de juegos 
# con descuentos o promocionaria juegos mas cortos, que les creen 
# curiosidad e interes en comprar mas juegos

####2: son el grupo medio, por ello, haria estrategias que los permitan 
# participar en juegos grupales, para que puedan invertir mas tiempo y 
# dinero comprando juegos y por pasar niveles adicionaria creditos en 
# sus cuentas para que de manera individual puedan comprar juegos

####3: ya que son los que mas compran, lo que haria es invitarlos a pasar 
# niveles que les den buenas recompensas que les ayude a seguir 
# comprando juego, sin embargo para lograrlo deben invertir bastante 
# tiempo, para tener los juegos con muy buenos descuentos

####4: una estrategia mixta como la del grupo 2, sin emabrgo en la que 
# compren mas y jueguen mas.

####5: una estrategia mixta como la del grupo 2, sin emabrgo en la que 
# compren mas y jueguen mas.

#2. Qué tipo de promociones recibiría cada segmento? Para los grupos en 
# que mas compran, haria promociones en las que tengan que invertir 
# tiempo jugando y les otorgan creditos para que puedan comprar mas 
# juegos en cuanto a los mas jugadores, daria premios pequeños y con 
# frdcuencia para que acumulen creditos y compren mas juegos

#3. Qué usuarios priorizarías para campañas de fidelización?
#Claro esta que haria mas campañas para obtener dinero, sin emabrgopara 
# ello deben jugar, y los beneficios pueden ser en sumar credito, en 
# diferentes cantidades dependiendo los juegos y la modalidad, por ello 
# dependiendo el desempaño de cada jugador puede comprar juegos con 
# diferente frecuencia

#4. Qué información adicional sería útil para mejorar la segmentación?
#como informacion adicional, seria bueno tener las categorias de los 
# juegos, las edades de los jugadores, la ubicacion e idioma de los 
# jugadores, las empresas que crean los juegos, los tipos de juegos.