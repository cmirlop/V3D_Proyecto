from Matriz_calibración_proyectada_P_et1 import main_p
from et2_factorizacion import main_f
from Matriz_Fundamental_F_Et3 import main_F
from Matriz_esencial_apartado_4 import calcular_matriz_essencial
import numpy as np
import cv2
import glob

image_paths = glob.glob('data/my_frame-*.jpg')
image_left = cv2.imread('data/left2.png') 
image_right = cv2.imread('data/right2.png')

#Fase 1 - Matriz de proyección de la cámara
matriz_p = main_p(image_paths)

#Fase 2 - Factorización de la matriz de proyección de la camara
K,R,t = main_f(matriz_p)

#Fase 3 - Matriz fundamental
F, pts_left, pts_right = main_F(image_left, image_right)

#Fase 4 - Matriz esencial
E = calcular_matriz_essencial(F, K)





