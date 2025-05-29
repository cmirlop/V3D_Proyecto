import calib
import factorizacion_P
import matriz_fundamental_F
import matriz_esencial_E
import numpy as np
from PIL import Image

rutas_imagenes = ["data/my_frame-00.jpg", "data/my_frame-01.jpg", "data/my_frame-02.jpg", "data/my_frame-03.jpg", "data/my_frame-04.jpg", "data/my_frame-05.jpg", "data/my_frame-07.jpg"]
tamano_tablero = (7, 5)
tamano_cuadro = 31
homografias = []

K = calib.calibracion(rutas_imagenes, tamano_tablero, tamano_cuadro, homografias)
print("Matriz de calibración K:", K)

homografias = np.array(homografias)

P = calib.matriz_p(K, homografias[0])
print("Matriz de proyección P:", P)

#------------------------------------------------------------------------------------

K, R = factorizacion_P.factorizacion_RQ(P)

print("Matriz K de parámetros intrínsecos:", K)
print("\nMatriz de Rotación R:", R)

t = factorizacion_P.taslacion(P, K)
print("\nVector de Traslación t:", t)

#------------------------------------------------------------------------------------

imagen1 = Image.open("data/im2.png").convert('L')
imagen2 = Image.open("data/im6.png").convert('L')

if imagen1.width > 800 or imagen2.width > 800:
    nuevo_tamano = (imagen1.width // 4, imagen1.height // 4)
    imagen1 = imagen1.resize(nuevo_tamano)
    imagen2 = imagen2.resize(nuevo_tamano)

imagen1 = np.array(imagen1)
imagen2 = np.array(imagen2)

C1 = matriz_fundamental_F.harris(imagen1, 0.4)
puntos1 = matriz_fundamental_F.puntos_harris(C1, 200)
C2 = matriz_fundamental_F.harris(imagen2, 0.4)
puntos2 = matriz_fundamental_F.puntos_harris(C2, 200)
matriz_fundamental_F.mostrar_puntos(imagen1, puntos1)
matriz_fundamental_F.mostrar_puntos(imagen2, puntos2)
parches1 = matriz_fundamental_F.extraer_parches(imagen1, puntos1)
parches2 = matriz_fundamental_F.extraer_parches(imagen2, puntos2)
coincidencias = matriz_fundamental_F.comparar_parches(parches1, parches2)
coincidencias = np.array(coincidencias)

puntos1 = puntos1[coincidencias[:, 0]]
puntos2 = puntos2[coincidencias[:, 1]]

F, inliers = matriz_fundamental_F.ransac(puntos1, puntos2, iteraciones=2000, umbral=0.5)
print("Matriz fundamental F:\n", F)

matriz_fundamental_F.visualizar_inliers(imagen1, imagen2, puntos1, puntos2, inliers)

np.save('matriz_F.npy', F)

#-------------------------------------------------------------------------------------

E = matriz_esencial_E.matriz_esencial_E(F, K)
print("Matriz esencial E:\n", E)

np.save('matriz_E.npy', E)

#-------------------------------------------------------------------------------------