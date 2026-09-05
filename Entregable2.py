import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\ASUS\Downloads\animal_data_dirty1.csv", sep=";")

print("1. Diagnóstico inicial")

#print(df.shape) #(1011, 1)
#print(df.columns) # Index(['Animal type;Country;Weight kg;Body Length cm;Gender;Animal code;Latitude;Longitude;Animal name;Observation date;Data compiled by'], dtype='str')
#print(df.info()) # la columna observation date y data compiled by son las unicas completas, en las demas hacen falta datos
#print(df.isna().sum()) #Animal type 20; Country 12;Weight kg 27;Body Length cm 27; Gender 19; Animal code 1011; Latitude 98; 
#Longitude 98; Animal name 959; Observation date 0; Data compiled by 0
#print(df.duplicated().sum()) #167

#Weight kg /  Body Length cm / Animal code / Latitude / Longitude / Observation date  (columnas float)
#Animal type / Country / Gender / Animal name  / Data compiled by  (columnas str)

#for columna in df.select_dtypes(include="object").columns:
#    print(df[columna].value_counts(dropna=False))

# se identifica en Animal type "red squirrel" y "red squirel", "lynx?" y "lynx", "European bison™" y "European bisson", "NaN"
# se identifica en Country que hay nombres completos y abreviatura de los paises, y datos NaN
#print(df.select_dtypes(include="number").describe())
# En las columnas numericas, se identifica que los valores minimos son negativos en peso y largo, lo cual no es informacion valida 
# los valores máximos tienen valores muy altos en ambas columnas por lo tanto es necesario verificarlos para identificar el origen de los datos
# ya que no son consistentes con la información promedio.


print("---------------------------------------------------")

print("2. Limpieza de datos")

# correccion de datos de tipo .str

df["Observation date"] = pd.to_datetime(
    df["Observation date"],
    errors="coerce"
)

df["Animal type"] = df["Animal type"].replace({
    "red squirrell": "Red Squirrel",
    "red squirel" : "Red Squirrel",
    "lynx?": "Lynx",
    "European bison™":"European bison",
    "European bisson":"European bison", 
    "ledgehod" : "Hedgehog",
    "wedgehod" : "Hedgehog"
})

df["Animal type"] = df["Animal type"].str.lower().str.strip().str.title()

df["Country"] = df["Country"].replace({
    "PL": "Poland",
    "HU": "Hungary",
    "CZ": "Czech Republic",
    "CC": "Czech Republic",
    "Czech": "Czech Republic",
    "DE": "Germany",
    "Hungry": "Hungary"
})

df["Country"] = df["Country"].str.lower().str.strip().str.title()

df["Gender"] = df["Gender"].str.lower().str.strip().str.title()

#filtrar los valores menores a 0 y revisar los datos para tener 
df = df[df["Body Length cm"] >= 0]
df = df[df["Weight kg"] >= 0]

#for columna in df.select_dtypes(include="object").columns:
#    print(df[columna].value_counts(dropna=False))

# correccion de datos de la columna ya que hay valores negativos

#for i in range(len(df["Body Length cm"])):
#    if df.loc[i, "Body Length cm"] < 0:
#        df.loc[i, "Body Length cm"] = -df.loc[i, "Body Length cm"]
#
#print(df["Body Length cm"].min())


#for i in range(len(df["Weight kg"])):
#    if df.loc[i, "Weight kg"] < 0:
#        df.loc[i, "Weight kg"] = -df.loc[i, "Weight kg"]

#print(df["Weight kg"].min())

#Se deja la seccion de "correccion de datos de la columna ya que hay valores negativos" en comentario ya que 
#con Numpy, puedo utilizarlos para continuar con las revisiones

print("---------------------------------------------------")

print("3. NumPy")

df["Weight problem"] = np.where(
    df["Weight kg"] < 0,
    "Valor incorrecto",
    "Correcto"
)

df["Length problem"] = np.where(
    df["Body Length cm"] < 0,
    "Valor incorrecto",
    "Correcto"
)

#Muestra los datos negativos
#print(df.loc[df["Weight problem"] == "Valor incorrecto", ["Weight kg", "Weight problem"]])
#print(df.loc[df["Length problem"] == "Valor incorrecto", ["Body Length cm", "Length problem"]])


for animal in df["Animal type"].dropna().unique():

    datos = df[df["Animal type"] == animal]

    #print("\n---", animal, "---")

    #print("Peso promedio:", np.nanmean(datos["Weight kg"]))
    #print("Peso mediana:", np.nanmedian(datos["Weight kg"]))
    #print("Peso desviación estándar:", np.nanstd(datos["Weight kg"]))

    #print("Longitud promedio:", np.nanmean(datos["Body Length cm"]))
    #print("Longitud mediana:", np.nanmedian(datos["Body Length cm"]))
    #print("Longitud desviación estándar:", np.nanstd(datos["Body Length cm"]))


print("---------------------------------------------------")

print("4. Análisis con groupby()")

print(df["Animal type"].value_counts()) #cantidad de datos
print(df.groupby("Animal type")["Weight kg"].mean()) #peso promedio por animal
print(df.groupby("Animal type")["Body Length cm"].mean()) #peso promedio por animal
print(df.groupby(["Country", "Animal type"])["Weight kg"].mean()) #promedio de peso por pais y animal
print(df.groupby(["Country", "Animal type"])["Body Length cm"].mean()) #promedio de longitud por pais y animal

print(df.groupby("Animal type")["Weight kg"].min()) # peso minimo por animal
print(df.groupby("Animal type")["Weight kg"].max()) # peso maximo por animal
print(df.groupby("Animal type")["Body Length cm"].min()) # longitud minima por animal
print(df.groupby("Animal type")["Body Length cm"].max()) # longitud maxima por animal


print("---------------------------------------------------")

print("5. Interpretación de resultados")

# El peso promedio del European Bison es de 592 kg. Al comparar por país, se observa que el promedio es mayor en Polonia, con 643.58 kg, mientras que en Slovakia es menor, con 219.55 kg.
# European Buster presenta un peso promedio de 535.25 kg. En Polonia el promedio aumenta a 604.5 kg, mientras que en Czech Republic disminuye a 466 kg.
# El peso promedio del Hedgehog es de aproximadamente 0.80 kg. Por país, el promedio es mayor en Polonia y menor en Australia, donde es de aproximadamente 0.70 kg.
# El peso promedio del Lynx es de 23.26 kg. Por país, el mayor promedio se encuentra en Polonia, con 27.84 kg, mientras que el menor se encuentra en Czech Republic, con 21.25 kg.
# Red Squirrel presenta un peso promedio de 0.29 kg. El promedio más alto se encuentra en Polonia, con 0.309 kg, y el más bajo en Germany, con 0.286 kg.
# Se observa que Polonia presenta, en general, los mayores promedios de peso entre los países analizados para las especies estudiadas.

# La longitud corporal promedio del European Bison es de 244.61 cm. En Polonia presenta un promedio mayor, de 258.33 cm, mientras que en Slovakia el promedio es menor, con 219.55 cm.
# El European Buster presenta una longitud corporal promedio de 238.75 cm. En Polonia el promedio es mayor, con 264.5 cm, mientras que en Czech Republic es menor, con 213 cm.
# El Hedgehog presenta una longitud corporal promedio de 20.22 cm. El promedio más alto se encuentra en Germany, con 22.17 cm, mientras que el menor se encuentra en Australia, con 19 cm.
# Lynx presenta una longitud corporal promedio de 82.9 cm. El promedio más alto se encuentra en Slovakia, con 87.38 cm, mientras que el menor se encuentra en Poland, con 79 cm.
# Red Squirrel presenta una longitud corporal promedio de 20.67 cm. El promedio más alto se encuentra en Poland, con 21.6 cm, mientras que el menor se encuentra en Hungary, con 19.69 cm.
