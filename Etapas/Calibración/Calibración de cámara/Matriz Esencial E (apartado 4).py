import numpy as np

# Cargar la matriz fundamental F desde el archivo exportado
def cargar_matriz_F():
    return np.load('matriz_F.npy')

# Cargar la matriz K desde el archivo exportado
def cargar_matriz_K():
    return np.load('matriz_K.npy')

# Función para calcular la matriz esencial (E)
def calcular_matriz_essencial(F, K):
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    S = np.array([1, 1, 0])  # Imponer rango 2
    E = U @ np.diag(S) @ Vt

    return E



#--- Pasos --- #

#1.- Cargar F exportada
F = cargar_matriz_F()

#2.- Cargar la matriz K exportada
K = cargar_matriz_K()

#3.- Calcular la matriz esencial E
E = calcular_matriz_essencial(F, K)

print("Matriz Esencial E:\n", E)
# Guardar la matriz Esencial en un archivo .npy
np.save('matriz_E.npy', E)