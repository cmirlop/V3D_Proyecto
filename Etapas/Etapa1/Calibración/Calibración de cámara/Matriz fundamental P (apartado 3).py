import numpy as np

# Función para normalizar los puntos
def normalizar_puntos(puntos):
    mean = np.mean(puntos, axis=0)
    std = np.std(puntos)
    S = np.sqrt(2) / std

    T = np.array([
        [S, 0, -S * mean[0]],
        [0, S, -S * mean[1]],
        [0, 0, 1]
    ])

    puntos_homog = np.hstack((puntos, np.ones((puntos.shape[0], 1))))
    puntos_normalizados = (T @ puntos_homog.T).T

    return puntos_normalizados[:, :2], T

# Función para calcular la matriz fundamental con 8 puntos
def calcular_F_8_puntos(puntos1, puntos2):
    puntos1, T1 = normalizar_puntos(puntos1)
    puntos2, T2 = normalizar_puntos(puntos2)

    A = np.zeros((len(puntos1), 9))

    for i in range(len(puntos1)):
        x1, y1 = puntos1[i]
        x2, y2 = puntos2[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]

    _, _, Vt = np.linalg.svd(A)
    F_normalizada = Vt[-1].reshape(3, 3)

    # Imponer rango 2
    U, S, Vt = np.linalg.svd(F_normalizada)
    S[2] = 0
    F_normalizada = U @ np.diag(S) @ Vt

    # Desnormalizar
    F = T2.T @ F_normalizada @ T1

    # Exportar la matriz F a un archivo para el apartado 4
    np.save("matriz_fundamental.npy", F)

    print("Matriz fundamental sin puntos:\n", F)

    return F / F[-1, -1]

# Ejemplo de uso
puntos1 = np.array([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100], [110, 120], [130, 140], [150, 160]])
puntos2 = np.array([[12, 22], [32, 42], [52, 62], [72, 82], [92, 102], [112, 122], [132, 142], [152, 162]])

F = calcular_F_8_puntos(puntos1, puntos2)

print("Matriz Fundamental F:\n", F)