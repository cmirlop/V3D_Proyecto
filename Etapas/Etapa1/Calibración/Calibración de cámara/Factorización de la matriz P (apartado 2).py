import numpy as np

# Cargar la matriz P guardada previamente
P = np.load('matriz_P.npy')

# Factorización de la matriz P usando el algoritmo Listing 4.3
def factorize_projection_matrix(P):
    # P es la matriz de proyección (3x4)
    M = P[:, :-1]  # Matriz 3x3 (las tres primeras columnas de P)
    q = P[:, -1]   # Vector de traslación (última columna de P)

    # Descomposición QR de la matriz M
    K, R = np.linalg.qr(np.linalg.inv(M))

    # Calcular el vector de traslación t
    t = np.linalg.inv(K) @ q

    # Asegurarnos de que K sea una matriz de parámetros intrínsecos
    K = np.linalg.inv(K)
    
    return K, R, t

# Realizar la factorización de la matriz P
K, R, t = factorize_projection_matrix(P)

print("Matriz K de parámetros intrínsecos:")
print(K)

print("\nMatriz de Rotación R:")
print(R)

print("\nVector de Traslación t:")
print(t)

# Guardar la matriz K en un archivo .npy
np.save('matriz_K.npy', K)
print("Matriz K guardada correctamente.")