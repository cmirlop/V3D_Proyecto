from Fase_1 import calib
from Fase_1 import factorizacion_P
from Fase_3 import matriz_fundamental_F
from Fase_3 import epipolares
from Fase_4 import matriz_esencial_E
from Fase_5 import rectificacion
from Fase_7 import Fase_7
import time
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

rutas_imagenes = ["Fase_1/data/my_frame-00.jpg", "Fase_1/data/my_frame-01.jpg", "Fase_1/data/my_frame-02.jpg", "Fase_1/data/my_frame-03.jpg", "Fase_1/data/my_frame-04.jpg", "Fase_1/data/my_frame-05.jpg", "Fase_1/data/my_frame-07.jpg"]
tamano_tablero = (7, 5)
tamano_cuadro = 31
homografias = []

'''
Esta parte se encarga al principio de cargar las imágenes para la calibración. Luego 
llama a las diferentes partes de la calibración, la cual se realiza mediante el algoritmo
de Z. Zhang. Lo que se obtiene de esta calibración es directaente la matriz de parámetros
intrínsecos K y de esta se puede sacar la matriz de proyección de la cámara P.
'''
# Utiliza las imágenes de calibración y los parámetros para calcular la matriz de calibración K
K = calib.calibracion(rutas_imagenes, tamano_tablero, tamano_cuadro, homografias)
print("Matriz de calibración K:", K)

# Guardamos las homografías en un array para poder utilizarlas posteriormente
homografias = np.array(homografias)

# Usamos la primera homografía para calcular la matriz de proyección P
P = calib.matriz_p(K, homografias[0])
print("Matriz de proyección P:", P)

#------------------------------------------------------------------------------------
'''
Esta parte se encarga de volver a sacar la matriz K para poder compararla con la anterior y
la matriz R de rotación. Para ello se utiliza la función de factorización RQ, además se saca
también el vector de traslación t.
'''
# Se utiliza la P anterior para factorizarla y obtener K, R y t para su comprobación
K, R = factorizacion_P.factorizacion_RQ(P)

print("Matriz K de parámetros intrínsecos:", K)
print("\nMatriz de Rotación R:", R)

t = factorizacion_P.taslacion(P, K)
print("\nVector de Traslación t:", t)

#------------------------------------------------------------------------------------
'''
Esta parte se encarga al principio de cargar las imágenes. Luego las redimensiona 
si son grandes, para así reducir el tiempo de computo. Tras haber redimensionado 
las imágenes se obienen los puntos de interés de las imágenes mediante el detector de Harris.
Luego se muestran los puntos de interés en las imágenes y se extraen los parches de las imágenes
correspondientes a los puntos de interés. Después se comparan los parches de las dos imágenes
y se obtienen las coincidencias. Por último, se aplica el algoritmo RANSAC para obtener la matriz 
fundamental F y se visualizan los inliers encontrados.
'''
# Carga las imágenes y las convierte a escala de grises
imagen1 = Image.open("Fase_7/data/izq3.png").convert('L')
imagen2 = Image.open("Fase_7/data/der3.png").convert('L')

# Redimensiona las imágenes si son grandes para reducir el tiempo de computo
if imagen1.width != 450 or imagen1.height != 375:
    nuevo_tamano = (450, 375)
    imagen1 = imagen1.resize(nuevo_tamano)
    imagen2 = imagen2.resize(nuevo_tamano)

# Convierte las imágenes a arrays numpy
imagen1 = np.array(imagen1)
imagen2 = np.array(imagen2)


# Obtiene los puntos de interés utilizando el detector de Harris
C1 = matriz_fundamental_F.harris(imagen1, 1)
puntos_izq_harris = matriz_fundamental_F.puntos_harris(C1, 0.08)
C2 = matriz_fundamental_F.harris(imagen2, 1)
puntos_der_harris = matriz_fundamental_F.puntos_harris(C2, 0.08)

# Ordena los puntos de interés por la respuesta del detector de Harris y selecciona los N mejores
N = 500
respuestas = C1[puntos_izq_harris[:,1], puntos_izq_harris[:,0]]
idx_orden = np.argsort(respuestas)[::-1][:N]
puntos_izq_harris = puntos_izq_harris[idx_orden]

respuestas = C2[puntos_der_harris[:,1], puntos_der_harris[:,0]]
idx_orden = np.argsort(respuestas)[::-1][:N]
puntos_der_harris = puntos_der_harris[idx_orden]

# Muestra los puntos de interés en las imágenes
matriz_fundamental_F.mostrar_puntos(imagen1, puntos_izq_harris)
matriz_fundamental_F.mostrar_puntos(imagen2, puntos_der_harris)

# Extrae los parches de las imágenes correspondientes a los puntos de interés
parches_izq = matriz_fundamental_F.extraer_parches(imagen1, puntos_izq_harris, tamano_parche=31)
parches_der = matriz_fundamental_F.extraer_parches(imagen2, puntos_der_harris, tamano_parche=31)

# Compara los parches de las dos imágenes y obtiene las coincidencias
coincidencias = matriz_fundamental_F.comparar_parches(parches_izq, parches_der, ratio=0.7)
coincidencias = np.array(coincidencias)

# Se sacan los puntos de la imágen izquierda y derecha respectivamente
puntos_izq_match = puntos_izq_harris[coincidencias[:, 0]]
puntos_der_match = puntos_der_harris[coincidencias[:, 1]]

matriz_fundamental_F.dibujar_coincidencias(imagen1, imagen2, puntos_izq_harris[coincidencias[:, 0]], puntos_der_harris[coincidencias[:, 1]])

# Se aplica el algoritmo RANSAC para obtener la matriz fundamental F y los inliers
F, inliers = matriz_fundamental_F.ransac(puntos_izq_match, puntos_der_match, iteraciones=1500, umbral=1, semilla=33)
print("Matriz fundamental F:\n", F)

# Visualiza los inliers encontrados en las imágenes
matriz_fundamental_F.visualizar_inliers(imagen1, imagen2, puntos_izq_match, puntos_der_match, inliers)

# Guarda la matriz fundamental F en un archivo .npy
np.save('matriz_F.npy', F)

#-------------------------------------------------------------------------------------
'''
Esta parte se encarga de utilizar la matriz fundamental F y la matriz de calibración K
para calcular la matriz esencial E.
'''
# Utiliza ls matrices fundamental F y K para calcular la matriz esencial E
E = matriz_esencial_E.matriz_esencial_E(F, K)
print("Matriz esencial E:\n", E)

# Guarda la matriz esencial E en un archivo .npy
np.save('matriz_E.npy', E)

#-------------------------------------------------------------------------------------
'''
Esta parte se encarga de mostrar las líneas epipolares en las imágenes utilizando la matriz fundamental F.
El funcionamiento es el siguiente: aparece una imagen con las dos imágenes originales una al lado de la otra,
en la imagen de la izquierda se pueden seleccionar puntos para dibujar las líneas epipolares en la otra imagen.
Una vez se cierra la ventana, emerge otra para hacer lo mismo pero esta ves seleccionando puntos en la imagen 
de la derecha y mostrando las líneas epipolares en la imagen de la izquierda.
'''

epipolares.dibujar_epipolar(imagen1, imagen2, F)
epipolares.dibujar_epipolar_inv(imagen1, imagen2, F)
# Solo los inliers
puntos_izq_inliers = puntos_izq_match[inliers]
puntos_der_inliers = puntos_der_match[inliers]
errores = epipolares.validar_epipolaridad(F, puntos_izq_inliers, puntos_der_inliers)
print("Errores de epipolaridad:", errores)

#-------------------------------------------------------------------------------------
'''
Esta parte se encarga al principio de cargar las imágenes. Luego las redimensiona 
si son grandes, para así reducir el tiempo de computo. Tras haber redimensionado 
las imágenes se obtienen las homografías rectificadas Hl y Hr a partir de los puntos
de interés. Luego se aplican las homografías a las imágenes originales para obtener 
las imágenes rectificadas y se muestran individualmente y luego en conjunto.
'''
# Carga las imágenes y las convierte a RGB para luego escalarlas
imagen1 = Image.open("Fase_7/data/izq3.png").convert('RGB')
imagen2 = Image.open("Fase_7/data/der3.png").convert('RGB')

if imagen1.width != 450 or imagen1.height != 375:
    nuevo_tamano = (450, 375)
    imagen1 = imagen1.resize(nuevo_tamano)
    imagen2 = imagen2.resize(nuevo_tamano)

# Se obtienen las homografías rectificadas Hl y Hr a partir de los puntos de interés
Hl, Hr = rectificacion.homografias_rectificadas(puntos_izq_match, puntos_der_match, puntos_izq_match[puntos_izq_match.shape[0] // 2], F)
print("Homografía izquierda Hl:\n", Hl)
print("Homografía derecha Hr:\n", Hr)

# Se obtienen las imágenes rectificadas aplicando las homografías a las imágenes originales
imagen_rectificada_izq = rectificacion.aplicar_homografia(imagen1, Hl)
imagen_rectificada_izq.show()
imagen_rectificada_dcha = rectificacion.aplicar_homografia(imagen2, Hr)
imagen_rectificada_dcha.show()

# Se muestran las imágenes rectificadas en conjunto
imagenes_rectificadas = rectificacion.dibujar_rectificaciones(imagen_rectificada_izq, imagen_rectificada_dcha)
imagenes_rectificadas.show()

#--------------------------------------------------------------------------------------
'''
Esta parte se encarga al principio de cargar las imágenes. Luego las redimensiona 
si son grandes, para así reducir el tiempo de computo. Tras haber redimensionado 
las imágenes se obtiene el mapa de disparidad, el cual se filtra y se obtienen sus
colores(valores RGB). Después se encarga de guardar el mapa de disparidad en un 
archivo PLY y genera una imagen con formato de mapa de calor, Por ultimo renderiza
en el visor 3D el mapa de disparidad con los colores que se acababa de guardar en un archivo PLY.
'''
# Carga las imagenes
left = Image.open("Fase_7/data/izq3.png")
right = Image.open("Fase_7/data/der3.png")

# Reduce el tamaño de las imagenes en caso de tener una anchura mayor a 800 para reducir tiempo de computo
if left.width != 450 and left.height != 375:
    new_size = (450, 375)
    left = left.resize(new_size)
    right = right.resize(new_size)

# Convierte las imagenes a escala de grises
left_gray = np.array(left.convert('L'))
right_gray = np.array(right.convert('L'))

# Obtiene la disparidad a partir de la imagen izquierda y derecha
start = time.time()
disparity = Fase_7.getDisparityMap(left_gray, right_gray)
end = time.time()

# Filtra la imagen y obtiene los colores de los pixeles de la imagen izquierda
disparity = Fase_7.median_blur(disparity, 5)
colors = np.array(left)

# Estadistica de tiempo de computo
print(f"Tiempo de generacion del mapa de disparidad: {end-start:.2f}s")

# Crea la carpeta en caso de no existir para almacenar el archivo de la nube de puntos y el mapa de calor correspondiente en formato PNG
if not os.path.exists("Fase_7/output"):
    os.mkdir("Fase_7/output")


Fase_7.compare_random_pixel(left_gray, right_gray, "Fase_7/output/SADvsSSDvsNCC.png") # Compara las distintas funciones de coste(SAD, SSD y NCC) en un pixel aleatorio
Fase_7.save_point_cloud(f"Fase_7/output/BM_python.ply", disparity, colors) # Guarda la nube de puntos en un archivo PLY
plt.imsave(f"Fase_7/output/BM_python.png", disparity, cmap='jet') # Guarda el mapa de calor de la imagen en base a la nube de puntos

# Muestra el resultado de la nube de puntos
Fase_7.render("Fase_7/output/BM_python.ply")