import numpy as np

# Cargar la matriz fundamental F desde el archivo exportado
def cargar_matriz_F():
    return np.load('matriz_fundamental.npy')

# Función para calcular la matriz esencial (E)
def calcular_matriz_essencial(F, K):
    E = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E)
    S = np.array([1, 1, 0])  # Imponer rango 2
    E = U @ np.diag(S) @ Vt

    return E

# Ejemplo de uso
puntos1 = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100], [110, 120], [130, 140], [150, 160]])
puntos2 = np.array([[12, 22], [32, 42], [52, 62], [72, 82], [92, 102], [112, 122], [132, 142], [152, 162]])

# Cargar F exportada
F = cargar_matriz_F()

K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])  # Utilización de los puntos
E = calcular_matriz_essencial(F, K)

print("Matriz Esencial E:\n", E)