import cv2
import numpy as np

#P = np.load('matriz_P.npy')

def factorizacion_RQ(P):
    M = P[:, :-1]  # Matriz 3x3 (las tres primeras columnas de P)
    m3 = M[2, :]
    m2 = M[1, :]

    q3 = m3 / np.linalg.norm(m3)

    q2 = (m2 - q3 * (np.dot(q3, m2))) / np.sqrt((np.linalg.norm(m2) ** 2) - (np.dot(q3, m2) ** 2))

    q1 = np.cross(q2, q3)

    Q = np.vstack((q1, q2, q3))

    U = M @ Q.T
    
    np.save('matriz_K.npy', U)
    np.save('matriz_R.npy', Q)

    return U, Q

def taslacion(P, K):
    t = np.linalg.inv(K) @ P[:, -1]   
    np.save('vector_t.npy', t)
    return t