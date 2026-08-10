print("Control de Flujo")

print("1. Registro de estudiantes")

estudiantes = [
    ("laura", 18, 4.5),
    ("carlos", 20, 3.8),
    ("andrea", 19, 4.2),
    ("juan", 21, 3.5),
    ("sofia", 18, 4.8),
    ("mateo", 22, 3.9),
    ("valentina", 20, 4.6),
    ("daniel", 19, 3.2),
]

print("2. Mostrar la información")

for i in estudiantes:
    print(f"{i[0]} tiene {i[1]} años y obtuvo una nota de {i[2]}")

print("3. Clasificación de estudiantes")

for i in estudiantes:
    if i[2] >= 4.5:
        print(f"{i[0]}: Excelente")
    elif i[2] >= 4.0 and i[2] <=  4.49:
        print(f"{i[0]}: Bueno")
    elif i[2] >= 3.0 and i[2] <=  3.99:
        print(f"{i[0]}: Aceptable")
    else:
        print(f"{i[0]}: Reprobó")

print("4. Promedio general")

suma = 0
conteo = 0

for i in estudiantes:
    suma += i[2]
    conteo += 1

promedio = suma / conteo
print(promedio)


print("5. Búsqueda de estudiante")

nombre = input("Escribe el nombre del estudiante: ")

esta = False

for i in estudiantes:
    if nombre == i[0]:
        esta = True
        break
       
if esta:
    print("El estudiante fue encontrado")
else:
    print("No se encontro ningun estudiante con ese nombre")

print("6. Diccionario de ciudades")

ubicacion = {
    "laura": "bogota",
    "carlos": "medellin",
    "andrea": "cali",
    "juan": "bogota",
    "sofia": "cartagena",
    "mateo": "medellin",
    "valentina": "barranquilla",
    "daniel": "cali"
}

for i in ubicacion:
    print(f"{i} vive en {ubicacion[i]}")

print("7. Cantidad de estudiantes por ciudad")

cantidad = {}

for i in ubicacion:
    ciudad = ubicacion[i]
    if ciudad in cantidad:
        cantidad[ciudad] += 1
    else:
        cantidad[ciudad] = 1

print(cantidad)

print("8. Ciclo While")

numero = float(input("Escribe un numero y 0 para finalizar: "))
sumatoria = 0
conteo = 0

while numero != 0:
    sumatoria += numero
    conteo += 1
    numero = float(input("Escribe un numero y 0 para finalizar: "))

if conteo > 0:
    promedio = sumatoria / conteo

    print(f"La suma de los numeros es: {sumatoria}")
    print(f"La cantidad de numeros escritos fue: {conteo}")
    print(f"El promedio de los numeros es: {promedio}")
else:
    print("No se han ingresado numeros")


print("9. Uso de break y continue")

for numeros in range(1, 31):
    if numeros % 3 == 0:
        continue
    if numeros == 25:
        break
    print(numeros)