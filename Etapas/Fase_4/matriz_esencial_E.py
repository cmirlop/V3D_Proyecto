import numpy as np

def matriz_esencial_E(F, K):
    E = K.T @ F @ K
    u, s, vt = np.linalg.svd(E)
    s = np.diag([1, 1, 0]) # se fuerza rango 2
    E = u @ s @ vt
    return E