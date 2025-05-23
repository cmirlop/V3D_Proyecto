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
    print("Matriz E antes de ajuste:\n", E)

    # Descomposición SVD
    U, S_vals, Vt = np.linalg.svd(E)

    print("\nValores singulares originales de E:", S_vals)

    # Corregimos si es necesario
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    # Reproyectamos E para que tenga rango 2 (como una matriz esencial ideal)
    S_fix = np.array([1, 1, 0])
    E = U @ np.diag(S_fix) @ Vt

    # Comprobamos los nuevos valores singulares
    _, S_fix_check, _ = np.linalg.svd(E)
    print("\nValores singulares ajustados de E:", S_fix_check)

    return E

#--- Pasos ---#

#1.- Cargar F exportada
F = cargar_matriz_F()

#2.- Cargar la matriz K exportada
K = cargar_matriz_K()

#3.- Calcular la matriz esencial E
E = calcular_matriz_essencial(F, K)

print("\nMatriz Esencial E final:\n", E)

# Guardar la matriz Esencial en un archivo .npy
np.save('matriz_E.npy', E)
