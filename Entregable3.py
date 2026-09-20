import kagglehub
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("nudratabbas/hospital-records-for-data-cleaning-medium")

archivos = os.listdir(path)

csv = [archivo for archivo in archivos if archivo.endswith('.csv')][0]

df = pd.read_csv(os.path.join(path, csv))


#-----------------------------------------------------------------------------------------------
# ----------------Identifica los primeros datos para conocer la BD -----------------------------

#print(df.head())
#    PatientID   Age   Gender              Diagnosis AdmissionDate DischargeDate HospitalID
#0  PN-2021066   7.0    Other  Myocardial Infarction    2024-03-23    2024-03-29    HOSP-65
#1  PN-4606019  36.0    Other              Pneumonia    2024-08-01    2024-08-07    HOSP-79
#2  PN-2594016  70.0    Other              Influenza    2024-11-16    2024-11-23    HOSP-27
#3  PN-6906914  90.0  Unknown       Acute Bronchitis    2025-07-05    2025-07-10    HOSP-64
#4  PN-4656204   0.0   Female        Type 2 Diabetes    2023-08-30    2023-08-31    HOSP-31

#-----------------------------------------------------------------------------------------------
#------------------------Identifica Filas y Columnas -------------------------------------------

#print(df.shape) 
# (5000, 7)  filas 5000 y columnas 7

#-----------------------------------------------------------------------------------------------
#---------------------Indica el nombre de las columnas de la BD --------------------------------

#print(df.columns)
#Index(['PatientID', 'Age', 'Gender', 'Diagnosis', 'AdmissionDate','DischargeDate', 'HospitalID'],dtype='str')

#-----------------------------------------------------------------------------------------------
#-------------------Indica el tipo de datos que hay en BD --------------------------------------

#print(df.dtypes)
#PatientID            str
#Age              float64
#Gender               str
#Diagnosis            str
#AdmissionDate        str  - Date
#DischargeDate        str  - Date
#HospitalID           str
#dtype: object

#-----------------------------------------------------------------------------------------------
#-------------------Identificar valores UNKNOWN en la BD-------- -------------------------------

df['Gender'] = df['Gender'].fillna('Unknown')
df['Diagnosis'] = df['Diagnosis'].fillna('Unknown')

print("Unknown en Gender:", (df['Gender'] == 'Unknown').sum())  #Unknown en Gender: 1513
print("Unknown en Diagnosis:", (df['Diagnosis'] == 'Unknown').sum()) #Unknown en Diagnosis: 350

#ya que son bastantes filas, es mejor no eliminarlos y tenerlos en cuenta en el analisis
#con el fin de mostrar la importancia de clasificar los datos de forma correcta y asi 
#ter un indicador futuro e informacion mas clara y precisa.

#-----------------------------------------------------------------------------------------------
#-------------------Cambiar el tipo de dato de la Columna --------------------------------------

df['AdmissionDate'] = pd.to_datetime(df['AdmissionDate'])
df['DischargeDate'] = pd.to_datetime(df['DischargeDate'])

#print(df.dtypes)

#PatientID                   str
#Age                     float64
#Gender                      str
#Diagnosis                   str
#AdmissionDate    datetime64[us]
#DischargeDate    datetime64[us]
#HospitalID                  str
#dtype: object

#-----------------------------------------------------------------------------------------------
#------------Indica cantidad de filas, si hay nulos, y tipo de dato ----------------------------

#print(df.info())

#<class 'pandas.DataFrame'>
#RangeIndex: 5000 entries, 0 to 4999
#Data columns (total 7 columns):
# #   Column         Non-Null Count  Dtype         
#---  ------         --------------  -----         
# 0   PatientID      5000 non-null   str           
# 1   Age            4650 non-null   float64          #hay valores nulos
# 2   Gender         4650 non-null   str              #hay valores nulos
# 3   Diagnosis      4650 non-null   str              #hay valores nulos
# 4   AdmissionDate  5000 non-null   datetime64[us]
# 5   DischargeDate  5000 non-null   datetime64[us]
# 6   HospitalID     5000 non-null   str           
#dtypes: datetime64[us](2), float64(1), str(4)
#memory usage: 450.8 KB
#None

#-----------------------------------------------------------------------------------------------
#--------------cuenta la cantidad de nulos por columna -----------------------------------------

#print(df.isnull().sum())

#PatientID          0
#Age              350
#Gender           350
#Diagnosis        350
#AdmissionDate      0
#DischargeDate      0
#HospitalID         0
#dtype: int64

#-----------------------------------------------------------------------------------------------
#----------- Da informacion sobre las columnas numericas ---------------------------------------
# max, min, 25%, 50%, 75%, media, mediana, conteo de datos, desviacion estandar ---------------- 

#print(df.describe())

#               Age               AdmissionDate               DischargeDate
#count  4650.000000                        5000                        5000
#mean     47.384301  2024-09-18 05:18:48.960000  2024-09-23 08:57:59.040000
#min       0.000000         2023-05-11 00:00:00         2023-05-12 00:00:00
#25%      23.000000         2024-01-13 00:00:00         2024-01-18 00:00:00
#50%      47.000000         2024-09-13 12:00:00         2024-09-19 00:00:00
#75%      72.000000         2025-05-26 00:00:00         2025-05-31 00:00:00
#max      95.000000         2026-02-12 00:00:00         2026-02-14 00:00:00
#std      27.880535                         NaN                         NaN

#Se identidica que el valor minimo es 0 el maximo 95

#-----------------------------------------------------------------------------------------------
#--------------- cuenta la cantidad de filas duplicadas ----------------------------------------

#print(df.duplicated().sum()) #0

#-----------------------------------------------------------------------------------------------
#-Ayuda a identificar si los valores nulos en las 3 columnas corresponden a los mismos pacientes

#print(df[df[['Age', 'Gender', 'Diagnosis']].isna().all(axis=1)])

#       PatientID  Age Gender Diagnosis AdmissionDate DischargeDate HospitalID
#4232  PN-8178896  NaN    NaN       NaN    2025-12-29    2026-01-07    HOSP-48
#4772  PN-6739983  NaN    NaN       NaN    2024-11-16    2024-11-20    HOSP-42

#Se identifica que solo 2 filas cumplen la condicion, por lo que son datos variados y al ser tanto no es 
#conveniente eliminar las filas relacionadas a los NaN

#-----------------------------------------------------------------------------------------------
#--------Identifica los valores unicos, ----------------------------------------------------

#print(df["Gender"].unique()) 
#['Other', 'Unknown', 'Female', 'Male', nan]
# Son diferentes no requieren de correccion

#print(df["Diagnosis"].unique()) 
#<ArrowStringArray>
#[  'Myocardial Infarction',               'Pneumonia',
#               'Influenza',        'Acute Bronchitis',
#         'Type 2 Diabetes',         'Gastroenteritis',
# 'Urinary Tract Infection',          'Diverticulitis',
#          'Cholelithiasis',          'CHOLELITHIASIS',
#            'HYPERTENSION',  'Chronic Kidney Disease',
#     'Atrial Fibrillation',                  'Asthma',
#                       nan,          'Osteoarthritis',
#            'Hypertension', 'URINARY TRACT INFECTION',
#               'INFLUENZA',         'GASTROENTERITIS',
#         'TYPE 2 DIABETES',                  'ASTHMA',
#               'PNEUMONIA',          'DIVERTICULITIS',
#        'ACUTE BRONCHITIS',  'CHRONIC KIDNEY DISEASE',
#     'ATRIAL FIBRILLATION',   'MYOCARDIAL INFARCTION',
#          'OSTEOARTHRITIS']
#Length: 29, dtype: str

# Se actualizan todos los nombres a mayuscula
df["Diagnosis"] = df["Diagnosis"].str.lower().str.strip()

#print(df["Diagnosis"].unique()) 

#[  'myocardial infarction',               'pneumonia',
#               'influenza',        'acute bronchitis',
#         'type 2 diabetes',         'gastroenteritis',
# 'urinary tract infection',          'diverticulitis',
#          'cholelithiasis',            'hypertension',
#  'chronic kidney disease',     'atrial fibrillation',
#                  'asthma',                       nan,
#          'osteoarthritis']
#Length: 15, dtype: str

#-----------------------------------------------------------------------------------------------
#----------La duracion de la hospitalizacion -  nueva columna-----------------------------------

df['Estancia'] = (
    df[['AdmissionDate', 'DischargeDate']].max(axis=1)
    - df[['AdmissionDate', 'DischargeDate']].min(axis=1)
).dt.days

df[['AdmissionDate', 'DischargeDate']].max(axis=1)
df[['AdmissionDate', 'DischargeDate']].min(axis=1)

#print(df['Estancia'])

#print(df.describe())

#               Age               AdmissionDate               DischargeDate                Estancia
#count  4650.000000                        5000                        5000                    5000
#mean     47.384301  2024-09-18 05:18:48.960000  2024-09-23 08:57:59.040000  5 days 03:39:10.080000
#min       0.000000         2023-05-11 00:00:00         2023-05-12 00:00:00       -5 days +00:00:00
#25%      23.000000         2024-01-13 00:00:00         2024-01-18 00:00:00         3 days 00:00:00
#50%      47.000000         2024-09-13 12:00:00         2024-09-19 00:00:00         5 days 00:00:00
#75%      72.000000         2025-05-26 00:00:00         2025-05-31 00:00:00         8 days 00:00:00
#max      95.000000         2026-02-12 00:00:00         2026-02-14 00:00:00        10 days 00:00:00
#std      27.880535                         NaN                         NaN  3 days 08:43:23.387987

# la maxima duracion en eshospitalizacion es de 10 dias y la minima es de 3 dias


#-----------------------------------------------------------------------------------------------
#----------------------------1.--Diagnósticos más frecuentes------------------------------------

#print(df['Diagnosis'].value_counts())

#Diagnosis
#urinary tract infection    363
#cholelithiasis             363
#osteoarthritis             352
#diverticulitis             345
#atrial fibrillation        345
#asthma                     344
#acute bronchitis           342
#influenza                  324
#gastroenteritis            320
#hypertension               319
#myocardial infarction      316
#chronic kidney disease     313
#pneumonia                  306
#type 2 diabetes            298
#Name: count, dtype: int64

#**************************************************************************
#***************************Conclusion*************************************

#Los diagnósticos más frecuentes son Urinary Tract Infection y Cholelithiasis, con 363 registros cada uno. 
#Por otro lado, el diagnóstico menos frecuente es Type 2 Diabetes, con un total de 298 registros.



#**************************************************************************
#***************************Grafico****************************************

diagnosticos = df['Diagnosis'].value_counts()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=diagnosticos.index,
    y=diagnosticos.values,
    palette="viridis"   
)

plt.title('Diagnósticos más frecuentes', fontsize=18,
    fontname='Comic Sans MS',
    fontweight='bold')
plt.xlabel('Diagnóstico', fontsize=14,
    fontname='Comic Sans MS')
plt.ylabel('Cantidad de pacientes', fontsize=14,
    fontname='Comic Sans MS')
plt.xticks(rotation=45, ha='right', fontsize=10, fontname='Comic Sans MS')
plt.yticks(fontsize=10, fontname='Comic Sans MS')
plt.tight_layout()

plt.show()

#-----------------------------------------------------------------------------------------------
#----------------------------2.--------Pacientes por hospital-----------------------------------

#print(df['HospitalID'].unique())
#<ArrowStringArray>
#['HOSP-65', 'HOSP-79', 'HOSP-27', 'HOSP-64', 'HOSP-31', 'HOSP-17', 'HOSP-68',
# 'HOSP-45', 'HOSP-39', 'HOSP-53', 'HOSP-72', 'HOSP-42', 'HOSP-47', 'HOSP-25',
# 'HOSP-98', 'HOSP-28', 'HOSP-84', 'HOSP-15', 'HOSP-60', 'HOSP-52', 'HOSP-21',
# 'HOSP-36', 'HOSP-48', 'HOSP-19', 'HOSP-49', 'HOSP-89', 'HOSP-54', 'HOSP-35',
# 'HOSP-43', 'HOSP-94', 'HOSP-62', 'HOSP-22', 'HOSP-73', 'HOSP-38', 'HOSP-51',
# 'HOSP-11', 'HOSP-71', 'HOSP-37', 'HOSP-66', 'HOSP-32', 'HOSP-90', 'HOSP-81',
# 'HOSP-33', 'HOSP-78', 'HOSP-24', 'HOSP-67', 'HOSP-96', 'HOSP-83', 'HOSP-44',
# 'HOSP-50', 'HOSP-85', 'HOSP-41', 'HOSP-46', 'HOSP-69', 'HOSP-63', 'HOSP-92',
# 'HOSP-10', 'HOSP-95', 'HOSP-26', 'HOSP-59', 'HOSP-91', 'HOSP-88', 'HOSP-70',
# 'HOSP-18', 'HOSP-16', 'HOSP-61', 'HOSP-23', 'HOSP-75', 'HOSP-87', 'HOSP-20',
# 'HOSP-57', 'HOSP-80', 'HOSP-58', 'HOSP-74', 'HOSP-56', 'HOSP-76', 'HOSP-93',
# 'HOSP-97', 'HOSP-99', 'HOSP-13', 'HOSP-55', 'HOSP-40', 'HOSP-29', 'HOSP-34',
# 'HOSP-77', 'HOSP-30', 'HOSP-86', 'HOSP-14', 'HOSP-12', 'HOSP-82']
#Length: 90, dtype: str

#print(df['HospitalID'].value_counts())
#HospitalID
#HOSP-25    72
#HOSP-48    71
#HOSP-32    68
#HOSP-73    67
#HOSP-23    66
#           ..
#HOSP-49    43
#HOSP-55    43
#HOSP-77    43
#HOSP-78    41
#HOSP-81    40
#Name: count, Length: 90, dtype: int64

#**************************************************************************
#***************************Conclusion*************************************

#El hospital que registra la mayor cantidad de pacientes es el HOSP-25, con un total de 72 pacientes. 
# Por otro lado, el hospital con menor cantidad de pacientes es el HOSP-81, con 40 pacientes.


#**************************************************************************
#***************************Grafico****************************************

pacientes_hospital = df['HospitalID'].value_counts().sort_index()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=pacientes_hospital.values,
    y=pacientes_hospital.index, 
    palette="summer"
)

plt.title('Número de pacientes por hospital', fontsize=18,
    fontname='Comic Sans MS',
    fontweight='bold')
plt.xlabel('Número de pacientes', fontsize=14,
    fontname='Comic Sans MS')
plt.ylabel('Hospital', fontsize=14,
    fontname='Comic Sans MS')
plt.xticks(rotation=45, ha='right', fontsize=10, fontname='Comic Sans MS')
plt.yticks(fontsize=3, fontname='Comic Sans MS')
plt.grid()
plt.tight_layout()

plt.show()

#-----------------------------------------------------------------------------------------------
#------------------3.-------Diagnósticos según género-------------------------------------------

#print(df.groupby(['Gender', 'Diagnosis']).size())
#Gender   Diagnosis              
#Female   acute bronchitis           69
#         asthma                     89
#         atrial fibrillation        83
#         cholelithiasis             92
#         chronic kidney disease     65
#         diverticulitis             84
#         gastroenteritis            70
#         hypertension               72
#         influenza                  89
#         myocardial infarction      74
#         osteoarthritis             80
#         pneumonia                  84
#         type 2 diabetes            57
#         urinary tract infection    84
#Male     acute bronchitis           86
#         asthma                     84
#         atrial fibrillation        77
#         cholelithiasis             87
#         chronic kidney disease     64
#         diverticulitis             70
#         gastroenteritis            58
#         hypertension               72
#         influenza                  59
#         myocardial infarction      81
#         osteoarthritis             70
#         pneumonia                  67
#         type 2 diabetes            77
#         urinary tract infection    75
#Other    acute bronchitis           82
#         asthma                     76
#         atrial fibrillation        70
#         cholelithiasis             86
#         chronic kidney disease     84
#         diverticulitis             95
#         gastroenteritis            88
#         hypertension               80
#         influenza                  73
#         myocardial infarction      81
#         osteoarthritis             81
#         pneumonia                  76
#         type 2 diabetes            74
#         urinary tract infection    77
#Unknown  acute bronchitis           81
#         asthma                     74
#         atrial fibrillation        86
#         cholelithiasis             74
#         chronic kidney disease     77
#         diverticulitis             74
#         gastroenteritis            80
#         hypertension               69
#         influenza                  75
#         myocardial infarction      63
#         osteoarthritis             90
#         pneumonia                  62
#         type 2 diabetes            74
#         urinary tract infection    98
#dtype: int64

#**************************************************************************
#***************************Grafico****************************************

tabla_genero = pd.crosstab(
    df['Gender'],
    df['Diagnosis']    
)

plt.figure(figsize=(10, 6))

sns.heatmap(
    tabla_genero,
    annot=True,
    fmt='d'
)

plt.title('Distribución de diagnósticos según género', fontsize=18,
    fontname='Comic Sans MS',
    fontweight='bold')
plt.xlabel('Diagnóstico', fontsize=14,
    fontname='Comic Sans MS')
plt.ylabel('Género', fontsize=14,
    fontname='Comic Sans MS')
plt.xticks(rotation=45, ha='right', fontsize=10, fontname='Comic Sans MS')
plt.yticks(fontsize=10, fontname='Comic Sans MS')
plt.tight_layout()

plt.show()  

#**************************************************************************
#***************************Conclusion*************************************

#En esta gráfica se puede obtener información relevante sobre la frecuencia de los diagnósticos según 
#el género. En el caso de las mujeres, el diagnóstico más común es la colelitiasis, mientras que el menos 
#frecuente es la diabetes tipo 2. En los hombres, la colelitiasis también es el diagnóstico más común, 
#mientras que la gastroenteritis y la influenza presentan la menor frecuencia, con una diferencia de solo 
#un registro entre ambas.

#Por otro lado, en la categoría de otro género es importante contar con registros más precisos. En esta categoría, 
#la diverticulitis presenta la mayor frecuencia, mientras que la fibrilación auricular registra la menor cantidad 
#de casos.

#Finalmente, en la categoría de género desconocido, es importante que el médico o la persona encargada del registro
#especifique correctamente esta información, ya que la gran cantidad de datos faltantes puede afectar el análisis. 
# Aun así, dentro de esta categoría se observa que los diagnósticos más frecuentes son la infección del tracto urinario 
#y la osteoartritis, mientras que los menos frecuentes son el infarto de miocardio y la neumonía.


#-----------------------------------------------------------------------------------------------
#---------------------4.---------cuántos pacientes hay en cada grupo----------------------------

df['GrupoEdad'] = pd.cut(
    df['Age'],
    bins=[0, 17, 30, 50, 70, float('inf')],
    labels=['0-17', '18-30', '31-50', '51-70', '71+']
)

#print(df['GrupoEdad'].value_counts().sort_index())

#GrupoEdad
#0-17      847
#18-30     622
#31-50     945
#51-70     975
#71+      1209
#Name: count, dtype: int64


#**************************************************************************
#***************************Grafico****************************************

pacientes_edad = df['GrupoEdad'].value_counts()

plt.figure(figsize=(10, 6))

sns.barplot(
    x=pacientes_edad.index,
    y=pacientes_edad.values, 
    palette="Wistia"
)

plt.title('Número de pacientes por grupo de edad', fontsize=18,
    fontname='Comic Sans MS',
    fontweight='bold')
plt.xlabel('Grupo de edad', fontsize=14,
    fontname='Comic Sans MS')
plt.ylabel('Número de pacientes', fontsize=14,
    fontname='Comic Sans MS')
plt.xticks(rotation=45, ha='right', fontsize=10, fontname='Comic Sans MS')
plt.yticks(fontsize=10, fontname='Comic Sans MS')
plt.tight_layout()

plt.show()


#**************************************************************************
#***************************Conclusion*************************************


#Los diagnósticos más frecuentes son Urinary Tract Infection y Cholelithiasis, con 363 registros cada uno. 
#En contraste, el diagnóstico menos frecuente es Type 2 Diabetes, con un total de 298 registros.



#-----------------------------------------------------------------------------------------------
#-----------5.--duración promedio de hospitalización según diagnóstico--------------------------

promedio_estancia = df.groupby('Diagnosis')['Estancia'].mean().sort_values()
#print(promedio_estancia)

#Diagnosis
#osteoarthritis            4 days 21:04:05.454545
#acute bronchitis          4 days 22:35:47.368421
#cholelithiasis            4 days 22:40:39.669421
#type 2 diabetes           5 days 01:31:48.724832
#gastroenteritis                  5 days 02:19:30
#diverticulitis            5 days 02:26:05.217391
#chronic kidney disease    5 days 03:17:49.648562
#pneumonia                 5 days 03:50:35.294117
#asthma                    5 days 04:32:05.581395
#influenza                 5 days 05:24:26.666666
#hypertension              5 days 05:56:36.865203
#urinary tract infection   5 days 07:12:23.801652
#myocardial infarction     5 days 07:49:22.025316
#atrial fibrillation       5 days 12:31:18.260869
#Name: Estancia, dtype: timedelta64[us]


#**************************************************************************
#***************************Conclusion*************************************

#El tiempo promedio de estancia por enfermedad es entre 4 a 5 dias

#**************************************************************************
#***************************Grafico****************************************

plt.figure(figsize=(10, 6))

sns.barplot(
    x=promedio_estancia.index,
    y=promedio_estancia.values, 
    palette="magma"
)

plt.title('Duración promedio de hospitalización según diagnóstico', fontsize=18,
    fontname='Comic Sans MS',
    fontweight='bold')
plt.xlabel('Diagnóstico', fontsize=14,
    fontname='Comic Sans MS')
plt.ylabel('Días promedio de hospitalización', fontsize=14,
    fontname='Comic Sans MS')
plt.xticks(rotation=45, ha='right', fontsize=10, fontname='Comic Sans MS')
plt.yticks(fontsize=10, fontname='Comic Sans MS')
plt.tight_layout()

plt.show()

#-----------------------------------------------------------------------------------------------
#-------------------6.--duración de hospitalización según grupo de edad-------------------------

#print(df.groupby('GrupoEdad', observed=True)['Estancia'].mean())

#GrupoEdad
#0-17    5 days 11:16:38.819362
#18-30   5 days 00:20:50.160771
#31-50   5 days 04:52:34.285714
#51-70   5 days 00:25:06.461538
#71+     5 days 02:30:04.466501
#Name: Estancia, dtype: timedelta64[us]


#**************************************************************************
#***************************Grafico****************************************

plt.figure(figsize=(10, 6))

promedio_estancia_edad = (
    df.groupby('GrupoEdad', observed=True)['Estancia']
      .mean()
)

sns.barplot(
    x=promedio_estancia_edad.index,
    y=promedio_estancia_edad.values, 
    palette="rainbow"
)

plt.title('Duración promedio de hospitalización según grupo de edad', fontsize=18,
    fontname='Comic Sans MS',
    fontweight='bold')
plt.xlabel('Grupo de edad', fontsize=14,
    fontname='Comic Sans MS')
plt.ylabel('Días promedio de hospitalización', fontsize=14,
    fontname='Comic Sans MS')
plt.xticks(rotation=45, ha='right', fontsize=10, fontname='Comic Sans MS')
plt.yticks(fontsize=10, fontname='Comic Sans MS')
plt.tight_layout()

plt.show()


#**************************************************************************
#***************************Conclusion*************************************

# El promedio de estancia segun el rango de edad es de 5 dias 


#-----------------------------------------------------------------------------------------------
#-----------------------7.----diagnosticos en los hospitales------------------------------------

diagnosticos_hospital = (
    df.groupby(['HospitalID', 'Diagnosis'])
      .size()
      .reset_index(name='Cantidad')
)


tabla_hospital_diagnostico = pd.crosstab(
    df['HospitalID'],
    df['Diagnosis']
)

plt.figure(figsize=(16, 12))

sns.heatmap(
    tabla_hospital_diagnostico,
    annot=True,
    fmt='d',
    cmap='Blues',
    annot_kws={"size": 5}
)

plt.title('Distribución de diagnósticos por hospital', fontsize=18,
    fontname='Comic Sans MS',
    fontweight='bold')
plt.xlabel('Diagnóstico', fontsize=14,
    fontname='Comic Sans MS')
plt.ylabel('Hospital', fontsize=14,
    fontname='Comic Sans MS')
plt.xticks(rotation=45, ha='right', fontsize=10, fontname='Comic Sans MS')
plt.yticks(fontsize=5, fontname='Comic Sans MS')

plt.tight_layout()
plt.show()

#En cuanto a los diagnósticos registrados, se observa que la bronquitis presenta mayor frecuencia en los hospitales
#30, 33, 51, 67 y 76. Por su parte, el asma se diagnostica principalmente en los hospitales 24, 68, 92 y 93, #mientras 
#que la fibrilación auricular se presenta en los hospitales 12 y 88.

#Asimismo, la colelitiasis se registra en los hospitales 33, 60, 61 y 96; la enfermedad renal crónica, en el 
#hospital 16; y la diverticulitis, en los hospitales 25 y 32. La gastroenteritis presenta registros en 
#los hospitales 17, 23, 57, 74 y 93, mientras que la hipertensión se concentra en los hospitales 20 y 24.

#En cuanto a otros diagnósticos, la influenza se registra en el hospital 62; el infarto de miocardio, en los 
#hospitales 63 y 68; la osteoartritis, en el hospital 26; la neumonía, en el hospital 46; y la 
#diabetes tipo 2, en los hospitales 57, 66 y 73. Finalmente, las infecciones del tracto urinario se 
#registran en el hospital 53.

#Cabe destacar que la categoría unknown aparece en todos los hospitales. Por esta razón, es importante prestar 
#especial atención a estos registros, con el fin de identificar correctamente los diagnósticos que actualmente se 
#encuentran clasificados como desconocidos."


#-----------------------------------------------------------------------------------------------
#---------------------------Conclusiones Finales------------------------------------------------

#como conclusiones finales, es posible establecer que el tiempo de estancia promedio independinete de la edad y 
# enfermedad es entre 4 y 5 dias

#como otra observacioin se puede identificar que debido a que la mayor cantidad de pacientes es mayor de 71 años

#infecciones del tracto urinario se #registran en el hospital 53 y es uno de los diagnosticos mas frecuentes y 
#la mayoria de estos registros no tienen un genero, lo que su correcto registro puede permitir identificar las causas 
#y crear estrategias para su mejora y prevencion

#en cuanto a los hospitales es posible identificar especialidades en los hospitales debido a la cantidad de pacientes 
#diagnosticados, al igual que establecer zonas, causas y origen de las enfermedades por el hospital, esto ayudara a
#identificar el tipo de medicos que requiere el hospital para un a efectiva atencion a los pacientes