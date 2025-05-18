import numpy as np

P = np.load('matriz_P.npy')

def Factorizazcion_algoritmo_Fusilleo(P):
    """
    Extrae los parámetros intrínsecos K, rotación R y traslación t a partir de la matriz de proyección P.
    """
    Q, U = np.linalg.qr(np.linalg.inv(P[:3, :3])) #Hacer QR de los 3X3 de P

    # Enforce negative focal lengths (signos)
    D = np.diag(np.sign(np.diag(U)) * np.array([-1, -1, 1]))

    Q = Q @ D
    U = D @ U

    # Asegurar que R tiene determinante positivo
    s = np.linalg.det(Q)
    R = s * Q.T
    t = s * U @ P[:3, 3]

    # Normalizar K tal que K[2,2] = 1
    K = np.linalg.inv(U / U[2, 2])

    return K, R, t

# Realizar la factorización de la matriz P
K, R, t = Factorizazcion_algoritmo_Fusilleo(P)
t = t.reshape((3, 1))

# Combina R y t en una matriz de 3x4
Rt = np.hstack((R, t))

# Matriz de proyección
P2 = K @ Rt

print(P)
print("sas")
print(P2)


print("Matriz K de parámetros intrínsecos:")
#print(K)
print("\nMatriz de Rotación R:")
#print(R)

print("\nVector de Traslación t:")
#print(t)

np.save('matriz_K.npy', K)