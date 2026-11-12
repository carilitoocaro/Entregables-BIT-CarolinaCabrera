import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix)

df = pd.read_csv(r"C:\Users\ASUS\Downloads\Bootcamp\Trabajos\archive\test_Y3wMUE5_7gLdaTN.csv")
df1 = pd.read_csv(r"C:\Users\ASUS\Downloads\Bootcamp\Trabajos\archive\train_u6lujuX_CVtuZ9i.csv") #(tiene Loan Status)

##########  Parte 1 – Comprensión del problema ##########

#print(df.head())
#    Loan_ID Gender Married Dependents     Education Self_Employed  ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History Property_Area
#0  LP001015   Male     Yes          0      Graduate            No             5720                  0       110.0             360.0             1.0         Urban
#1  LP001022   Male     Yes          1      Graduate            No             3076               1500       126.0             360.0             1.0         Urban
#2  LP001031   Male     Yes          2      Graduate            No             5000               1800       208.0             360.0             1.0         Urban
#3  LP001035   Male     Yes          2      Graduate            No             2340               2546       100.0             360.0             NaN         Urban
#4  LP001051   Male      No          0  Not Graduate            No             3276                  0        78.0             360.0             1.0         Urban
#print(df1.head())
#    Loan_ID Gender Married Dependents     Education Self_Employed  ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History Property_Area Loan_Status
#0  LP001002   Male      No          0      Graduate            No             5849                0.0         NaN             360.0             1.0         Urban           Y
#1  LP001003   Male     Yes          1      Graduate            No             4583             1508.0       128.0             360.0             1.0         Rural           N
#2  LP001005   Male     Yes          0      Graduate           Yes             3000                0.0        66.0             360.0             1.0         Urban           Y
#3  LP001006   Male     Yes          0  Not Graduate            No             2583             2358.0       120.0             360.0             1.0         Urban           Y
#4  LP001008   Male      No          0      Graduate            No             6000                0.0       141.0             360.0             1.0         Urban           Y

#-------Identifica la variable objetivo (target) -------
# df1["Loan_Status"]

#-------Describe brevemente el contexto del problema -------
#Luego de revisar en detalle los 5 datos iniciales de cada una de las BD, es posible identificar que ambas tienen informacion rrelacionada con prestamos, 
#donde se puede ver el codigo del prestamo que se solicita, el genero de quien lo solocota, si iest acasado, si tiene personas que dependan de esta, 
#si es una pesona graduada, si es empleado, ingresos de quien hace la solicitud y los ingresos del codeudor, el valor que solicita, el plazo maximo de pago, 
#el historial crediticio, si esta ubicado en la zona urbana o rural. 
#Las 2 bases de datos contienen la misma informacion, sin embargo df1 tiene una adicional que es Loan_Status que seria la BD de entrenamiento del sistema para 
#aplicar el aprendizaje supervisado.

##########  Parte 2 – Exploración de datos (EDA básica) ##########

#print(df.info())
# #   Column             Non-Null Count  Dtype  
#---  ------             --------------  -----  
# 0   Loan_ID            367 non-null    str    
# 1   Gender             356 non-null    str    
# 2   Married            367 non-null    str    
# 3   Dependents         357 non-null    str    
# 4   Education          367 non-null    str    
# 5   Self_Employed      344 non-null    str    
# 6   ApplicantIncome    367 non-null    int64  
# 7   CoapplicantIncome  367 non-null    int64  
# 8   LoanAmount         362 non-null    float64
# 9   Loan_Amount_Term   361 non-null    float64
# 10  Credit_History     338 non-null    float64
# 11  Property_Area      367 non-null    str 
#print(df.describe())
#       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History
#count       367.000000         367.000000  362.000000        361.000000      338.000000
#mean       4805.599455        1569.577657  136.132597        342.537396        0.825444
#std        4910.685399        2334.232099   61.366652         65.156643        0.380150
#min           0.000000           0.000000   28.000000          6.000000        0.000000
#25%        2864.000000           0.000000  100.250000        360.000000        1.000000
#50%        3786.000000        1025.000000  125.000000        360.000000        1.000000
#75%        5060.000000        2430.500000  158.000000        360.000000        1.000000
#max       72529.000000       24000.000000  550.000000        480.000000        1.000000
#print(df.columns)
#['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#       'Loan_Amount_Term', 'Credit_History', 'Property_Area']
#print(df.shape)
#(367, 12)
#print(df.isnull().sum())
#Loan_ID               0
#Gender               11
#Married               0
#Dependents           10
#Education             0
#Self_Employed        23
#ApplicantIncome       0
#CoapplicantIncome     0
#LoanAmount            5
#Loan_Amount_Term      6
#Credit_History       29
#Property_Area         0
#print(df.duplicated().sum()) #0

#+++++ Correccion de Valores nulos +++++

df["Gender"] = df["Gender"].fillna("No especificado")
df["Dependents"] = df["Dependents"].fillna("No especificado")
df["Self_Employed"] = df["Self_Employed"].fillna("No especificado")

df["LoanAmount"] = df["LoanAmount"].astype("Float64")
df["Loan_Amount_Term"] = df["Loan_Amount_Term"].astype("Float64")
df["Credit_History"] = df["Credit_History"].astype("Float64")

#+++++ Ver valores unicos por columna para identificar errores +++++

#print(df["Gender"].unique())
#['Male', 'Female', 'No especificado']
#print(df["Married"].unique())
#['Yes', 'No']
#print(df["Dependents"].unique())
#['0', '1', '2', '3+', 'No especificado']
#print(df["Education"].unique())
#['Graduate', 'Not Graduate']
#print(df["Self_Employed"].unique())
#['No', 'Yes', 'No especificado']
#print(df["Loan_Amount_Term"].unique())
#[360.0, 240.0, 180.0,  <NA>,  60.0, 480.0,  84.0,  12.0, 300.0, 350.0,  36.0,
# 120.0,   6.0]
#print(df["Credit_History"].unique())
#[1.0, <NA>, 0.0]
#print(df["Property_Area"].unique())
#['Urban', 'Semiurban', 'Rural']

#--------------------------------------------------------------------------------------
#print(df1.info())
# #   Column             Non-Null Count  Dtype  
#---  ------             --------------  -----  
# 0   Loan_ID            614 non-null    str    
# 1   Gender             601 non-null    str    
# 2   Married            611 non-null    str    
# 3   Dependents         599 non-null    str    
# 4   Education          614 non-null    str    
# 5   Self_Employed      582 non-null    str    
# 6   ApplicantIncome    614 non-null    int64  
# 7   CoapplicantIncome  614 non-null    float64
# 8   LoanAmount         592 non-null    float64
# 9   Loan_Amount_Term   600 non-null    float64
# 10  Credit_History     564 non-null    float64
# 11  Property_Area      614 non-null    str    
# 12  Loan_Status        614 non-null    str 
#print(df1.describe())
#       ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  Credit_History
#count       614.000000         614.000000  592.000000         600.00000      564.000000
#mean       5403.459283        1621.245798  146.412162         342.00000        0.842199
#std        6109.041673        2926.248369   85.587325          65.12041        0.364878
#min         150.000000           0.000000    9.000000          12.00000        0.000000
#25%        2877.500000           0.000000  100.000000         360.00000        1.000000
#50%        3812.500000        1188.500000  128.000000         360.00000        1.000000
#75%        5795.000000        2297.250000  168.000000         360.00000        1.000000
#max       81000.000000       41667.000000  700.000000         480.00000        1.000000
#print(df1.columns)
#['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
#       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
#       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status']
#print(df1.shape)
#(614, 13)
#print(df1.isnull().sum())
#Loan_ID               0
#Gender               13
#Married               3
#Dependents           15
#Education             0
#Self_Employed        32
#ApplicantIncome       0
#CoapplicantIncome     0
#LoanAmount           22
#Loan_Amount_Term     14
#Credit_History       50
#Property_Area         0
#Loan_Status           0
#print(df1.duplicated().sum()) #0

#+++++ Correccion de Valores nulos +++++

#Tratamiento de valores nulos
#Con el fin de manatener los datos dentro del rango sin necesidad de elimar las filas afectadas con este tipo de valores
#Se toman las siguientes decisiones: 
# Numéricas: imputar con la mediana.

df["ApplicantIncome"] = df["ApplicantIncome"].fillna(df["ApplicantIncome"].mean())

df["CoapplicantIncome"] = df["CoapplicantIncome"].fillna(df["CoapplicantIncome"].mean())

df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].mean())

# Categóricas: se dejan los valores como "No especificado", para tener una visual clara del estado de los valores nulos segun 
# cada columna.

df1["Gender"] = df1["Gender"].fillna("No especificado")
df1["Married"] = df1["Married"].fillna("No especificado")
df1["Dependents"] = df1["Dependents"].fillna("No especificado")
df1["Self_Employed"] = df1["Self_Employed"].fillna("No especificado")

#+++++ Ver valores unicos por columna para identificar errores +++++
#print(df1["Gender"].unique())
#['Male', 'Female', 'No especificado']
#print(df1["Married"].unique())
#['Yes', 'No', nan]
#print(df1["Dependents"].unique())
#['0', '1', '2', '3+', 'No especificado']
#print(df1["Education"].unique())
#['Graduate', 'Not Graduate']
#print(df1["Self_Employed"].unique())
#['No', 'Yes', 'No especificado']
#print(df1["Loan_Amount_Term"].unique())
#[360.0, 120.0, 240.0, <NA>, 180.0, 60.0, 300.0, 480.0, 36.0, 84.0, 12.0]
#print(df1["Credit_History"].unique())
#[1.0, <NA>, 0.0]
#print(df1["Property_Area"].unique())
#['Urban', 'Semiurban', 'Rural']

#+++++ Graficos +++++

# A) Cantidad de prestamos aprobados y rechazados

#plt.figure(figsize=(7,5))

#sns.countplot(
#    data=df1,
#    x="Loan_Status",
#    hue="Loan_Status",
#    palette="Set2",
#    legend=False
#)

#plt.title("Distribución del estado de los préstamos")
#plt.xlabel("Estado del préstamo")
#plt.ylabel("Cantidad de solicitantes")

#plt.grid(axis="y")
#plt.show()

#se identifica qye la cantidad de creditos aprobados es de mas de 400 y los que se han rechazado son casi 200 de los 614 registros de la BD

#B) Relacion entre ingresos y monto solicitado, teniendo en cuenta los pretamos aprobados y los rechazados

#plt.figure(figsize=(8,6))

#sns.scatterplot(
#    data=df1,
#    x="ApplicantIncome",
#    y="LoanAmount",
#    hue="Loan_Status",
#    alpha=0.7
#)

#plt.title("Ingresos vs monto del préstamo")
#plt.xlabel("Ingresos del solicitante")
#plt.ylabel("Monto del préstamo")
#plt.legend(title="Loan Status")

#plt.show()

#La mayor concentracion de los datos se encuentra entre 0 y 20.000 con lo referente a los ingresos del solicitante.
#En cuanto a al monto solicitado del prestamo es posible identificar que van desde valores cercanos a 0 y valores cerca de 300
#de la misma manera la mayor cantidad de registros en este espacio han sido aprobados

#C) Correlacion entre las columnas numericas

#columnas = [
#    "ApplicantIncome",
#    "CoapplicantIncome",
#    "LoanAmount",
#    "Loan_Amount_Term",
#    "Credit_History"
#]

#correlacion = df1[columnas].corr()

#plt.figure(figsize=(8,6))

#sns.heatmap(
#    correlacion,
#    annot=True,
#    cmap="coolwarm",
#    fmt=".2f",
#    linewidths=0.5,
#    vmin=-1,
#    vmax=1,
#    center=0
#)

#plt.title("Correlación entre variables numéricas")

#plt.show()

#Cercano a 1: correlación positiva.
#Cercano a -1: correlación negativa.
#Cercano a 0: poca relación lineal.

#las variables nuemricas no estan relacionadas entre si, sin embargoo la que esta mas cercana a tener
#una relacion es ApplicantIncome y LoanAmount con un 0.57 de correlacion positiva
# En cuanto a las demas relaciones, no hay alguna que sea significativa sobre la aprobacion de los prestamos

#D) tasa de aprobación según el historial crediticio

#plt.figure(figsize=(8, 5))

#sns.countplot(
#    data=df1,
#    x="Credit_History",
#    hue="Loan_Status",
#    palette="Set2"
#)

#plt.title("Estado del préstamo según el historial crediticio")
#plt.xlabel("Historial crediticio")
#plt.ylabel("Cantidad de solicitantes")

#plt.xticks([0, 1], ["Sin historial favorable", "Historial favorable"])
#plt.legend(title="Estado del préstamo",
#           labels=["Rechazado", "Aprobado"])

#plt.show()

# Es posible ver que cuando un solicitante de prestamo tiene un historial crediticio favorable tiene una mayor posibilidad de que el prestamo sea aprobado, 
#esto debido a que en la grafica se puede ver que se han aprobado mas de 350 prestamos cuando el historial es favorale y se han rechazado cerca de 100
#Por otra parte cuando el historial no es favorable se han rechazado cerca de 80 y se han aprobado cerca de 10

##########  Parte 3 – Preparación de datos ##########

#Definicion de X y y

X = df1.drop(columns=["Loan_ID", "Loan_Status"])
y = df1["Loan_Status"].map({"Y": 1,"N": 0})

#Columnas categoricas
categoricas = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area"
]

#Columnas numericas
numericas = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

# valores nulos
#categoricos: utilizo la moda, para que los valores nulos no se deban eliminar y para que estos no alteren de forma considerable los resultados
Nulos_categoricos = Pipeline(steps=[
    ("imputacion", SimpleImputer(strategy="most_frequent")),
    ("codificacion", OneHotEncoder(handle_unknown="ignore"))
])

#numericos: utilizo la mediana, para que los valores nulos no se deban eliminar y para que estos no alteren de forma considerable los resultados
Nulos_numericos = Pipeline(steps=[
    ("imputacion", SimpleImputer(strategy="median")),
    ("escalado", StandardScaler())
])

#Integracion de los valores nulos

preparacion_bd = ColumnTransformer(
    transformers=[
        ("numericas", Nulos_numericos, numericas),
        ("categoricas", Nulos_categoricos, categoricas)
    ]
)

#Dividir los datos: Organizar datos para el trainig y el test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

#print(X.shape) #(614, 11)
#print(y.shape) #(614,)

#COLUMNAS DE X
#print(X.columns.tolist())
#['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']

#VALORES DE y
#print(y.value_counts(dropna=False))
#Loan_Status
#1    422
#0    192

#PRIMEROS REGISTROS DE X
#print(X.head())
#  Gender Married Dependents  ... Loan_Amount_Term Credit_History  Property_Area
#0   Male     Yes          0  ...            360.0            1.0          Urban
#1   Male     Yes          1  ...            360.0            1.0          Rural
#2   Male     Yes          0  ...            360.0            1.0          Urban
#3   Male     Yes          0  ...            360.0            1.0          Urban
#4   Male      No          0  ...            360.0            1.0          Urban

#PRIMEROS REGISTROS DE y
#print(y.head())
#0    1
#1    0
#2    1
#3    1
#4    1

##########  Parte 4 – Modelado ##########

#A) Regresion Logistica 

modelo_lr = Pipeline(steps=[
    ("preprocesamiento", preparacion_bd),
    ("clasificador", LogisticRegression(max_iter=1000))
])

modelo_lr.fit(X_train, y_train)

predicciones_lr = modelo_lr.predict(X_test)

#Regresion Logistica - Resultados
#print(modelo_lr.named_steps["clasificador"])
#LogisticRegression(max_iter=1000) #Esta bien configurado

#print(modelo_lr.named_steps["clasificador"].classes_)
#[0 1]  reconoce los valores que debe predecir

#B) Random Forest SVC

modelo_svm = Pipeline(steps=[
    ("preprocesamiento", preparacion_bd),
    ("clasificador", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    ))
])

modelo_svm.fit(X_train, y_train)

predicciones_svm = modelo_svm.predict(X_test)

#SUPPORT VECTOR MACHINE (SVC)- Resultados
#print(modelo_svm.named_steps["clasificador"])
#SVC(class_weight='balanced')  #Esta bien configurado
#print(modelo_svm.named_steps["clasificador"].classes_)
#[0 1]  reconoce los valores que debe predecir


##########  Parte 5 – Evaluación del modelo ##########

#A) función de evaluación

def evaluar_modelo(nombre, y_real, y_predicho):
    accuracy = accuracy_score(y_real, y_predicho)
    precision = precision_score(
        y_real,y_predicho,zero_division=0
        )

    recall = recall_score(
        y_real,y_predicho,zero_division=0
    )

    f1 = f1_score(
        y_real,y_predicho,zero_division=0
    )

    #print(f"\n===== {nombre} =====")

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    matriz = confusion_matrix(
        y_real,
        y_predicho,
        labels=[0, 1]
    )

    print(matriz)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rechazado", "Aprobado"],
        yticklabels=["Rechazado", "Aprobado"]
    )

    plt.title(f"Matriz de confusión - {nombre}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")

    plt.show()

    return {
        "Modelo": nombre,
        "Accuracy": accuracy,
       "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

#Resultados de ambos modelos

resultados_lr = evaluar_modelo(
    "Regresión Logística",
    y_test,
    predicciones_lr
)

resultados_svm = evaluar_modelo(
    "SVM",
    y_test,
    predicciones_svm
)

comparacion = pd.DataFrame([
    resultados_lr,
    resultados_svm
])

#print(comparacion)

#Regresion Logistica
#Accuracy: 0.8618 
#acierta en un 86.18% en las predicciones
#Precision: 0.84 
#cuando predice un préstamo aprobado acierta en el 84% de las veces
#Recall: 0.9882
#identifica el 98.82% de los prestamos que realmente son aprobados
#F1-score: 0.9081
#Debido a que el procentaje es mas alto en RL demuestra que tiene un mejor equilibrio entre la precision y el Recall al predecir los aprobados
#matriz de confusion
#[[22 16]
# [ 1 84]]
#identifico 22 valores correctos como rechazados
#identifico 16 rechazados que eran correctos
#identifico 1 prestamo aprobado como rechazado
#identifico 84 aprobados correctos

#SVM
#Accuracy: 0.8455 
#acierta en un 84.55% en las predicciones
#Precision: 0.8511 
#cuando predice un préstamo aprobado acierta en el 85.11% de las veces
#Recall: 0.9412
#identifica el 94.12% de los prestamos que realmente son aprobados
#F1-score: 0.8939
#matriz de confusion
#[[24 14]
# [ 5 80]]

#identifico 24 valores correctos como rechazados
#identifico 14 rechazados que eran correctos
#identifico 5 prestamo aprobado como rechazado
#identifico 80 aprobados correctos

#B) Evaluar Regresión Logística

resultados_lr = evaluar_modelo(
    "Regresión Logística",
    y_test,
    predicciones_lr
)

#C) Evaluar SVM

resultados_svm = evaluar_modelo(
    "Support Vector Machine",
    y_test,
    predicciones_svm
)

#D) Comparación de resultados

comparacion = pd.DataFrame([
    resultados_lr,
    resultados_svm
])

#print(comparacion) #Compactar resultados de las metricas excepto la matriz de confusion 

#                   Modelo  Accuracy  Precision    Recall  F1-score
#0     Regresión Logística  0.861789   0.840000  0.988235  0.908108
#1  Support Vector Machine  0.845528   0.851064  0.941176  0.893855

#####---------------Seleccionar el mejor modelo

mejor_modelo_nombre = comparacion.loc[
    comparacion["F1-score"].idxmax(),
    "Modelo"
]

if mejor_modelo_nombre == "Regresión Logística":
    mejor_modelo = modelo_lr
else:
    mejor_modelo = modelo_svm

print("Mejor modelo:", mejor_modelo_nombre)


##########  Parte 6 – Interpretación ##########

#1. ¿Qué tan bueno es el modelo? 
# luego de revisar las metricas generadas por ambos modelos por separado, es posible identificar que la regresion logistica,
# es la mejor para predecir los prestamos aprobados ya que tiene un rendimiento del 86.18%, un f1-score de 90.81%, sin embargo, 
# esto no quiere decir que tenga una alta precision en los rechazados o que el modelo funcione a la perfeccion, por ello, es 
# importante seguir adicionando informacion en la BD para que la prediccion cada vez sea mejor al momento de predecir si un 
# prestamo sera o no aprobado, teniendo la informacion suficiente. 

#2. ¿Qué variable parece influir más en la predicción?
 
#Como se vio al inicio es posible identificar que la mayor relacion en las variables que influyan en la aprobacion del prestamo 
# estan entre el valor del prestamo, los ingresos del solicitante y del historial creditio de quien solicita el prestamo, ya que 
# si la persina tiene un buen historial credicio, tiene mayor posibilidad de que este sea aprobado.