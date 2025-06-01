import numpy as np
from scipy.ndimage import affine_transform

def epipolo(F):
    _, _, Vt = np.linalg.svd(F)
    e = Vt[-1]
    return e / e[2]

def matriz_M(F):
    # Descomponemos la matriz F
    u, s, vt = np.linalg.svd(F)

    # Calculamos el valor de lambda
    lam = (s[0] + s[1]) / 2

    # Modificamos los valores singulares
    S = np.array([[0, s[1], 0],
                  [-s[0], 0, 0],
                  [0, 0, lam]])

    # Recomponemos para obtener la matriz M
    M = u @ S @ vt
    return M

def homografias_rectificadas(puntos_izq, puntos_dcha, punto, F):
    # Obtenemos el epipolo izquierdo
    e_izq = epipolo(F)

    # Obtenemos la matriz M
    M = matriz_M(F)

    # Obtenemos la matriz de transformación T
    T_trans = np.array([[1, 0, -punto[0]],
                   [0, 1, -punto[1]],
                   [0, 0, 1]])
    
    # Sacamos el epipolo trasladado
    e_izq_trasladado = T_trans @ e_izq

    # Con el epipolo trasladado sacamos la matriz de rotacion
    alpha = np.arctan2(e_izq_trasladado[1], e_izq_trasladado[0])
    if alpha > np.pi / 2:
        alpha += np.pi
    T_rot = np.array([[np.cos[alpha], np.sin[alpha], 0],
                   [-np.sin[alpha], np.cos[alpha], 0],
                   [0, 0, 1]])
    
    # Sacamos el epipolo rectificado ahora
    e_izq_rectificado = T_rot @ e_izq_trasladado

    # Ahora sacmos la homografia al infinito de la imagen izquierda
    Hinf = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [-1/e_izq_rectificado[0], 0, 1]])
    
    # Sacamos la homografia de la imagen izquierda
    Hl = Hinf @ T_rot @ T_trans

    # Transformamos las imagenes
