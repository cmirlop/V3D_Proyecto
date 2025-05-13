
import numpy as np


def rectificacion_Esteroscipica_calibrada(E,y1,y2):

    #Obtenemos SVD
    U,S,Vt = np.linalg.svd(E)

    #Creamos la matriz W
    W = np.array([0,1,0],[-1,0,0],[0,0,1])

    #Poses camara
    t_hat = U[:, 2]
    P1 = [Vt.T @ W @ U.T, t_hat]
    P2 = [Vt.T @ W.T @ U.T, t_hat]
    P3 = [Vt.T @ W @ U.T, t_hat]
    P4 = [Vt.T @ W.T @ U.T, t_hat]