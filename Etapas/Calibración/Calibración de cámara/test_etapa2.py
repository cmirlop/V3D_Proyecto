import numpy as np
import unittest
from et2_factorizacion import Factorizazcion_algoritmo_Fusilleo


class TestFactorizacion(unittest.TestCase):

    #Realizamos la recosntrucción de P
    def test_reconstruccion_P(self):
        P = np.load('matriz_P.npy')
        R = np.load('matriz_R_et2.npy')
        K = np.load('matriz_K.npy')
        t = np.load('matriz_t_et2.npy')

        t = t.reshape((3, 1))

        # Combina R y t en una matriz de 3x4
        Rt = np.hstack((R, t))

        # Matriz de proyección
        P2 = K @ Rt
    
        P = P / P[-1, -1]
        P2 = P2 / P2[-1, -1]

        self.assertTrue(np.allclose(P,P2, atol=1e-6))
    
    #Comprobamos que R @ R.T es igual a la identidad
    def test_R_es_ortogonal(self):
        R = np.load('matriz_R_et2.npy')
        I = np.eye(3)
        self.assertTrue(np.allclose(R @ R.T, I, atol=1e-6))

    #COmprobamos que el determinante es 1
    def test_determinante_R(self):
        R = np.load('matriz_R_et2.npy')
        # Determinante de la rotación
        det_R = np.linalg.det(R)

        print(f"Determinante de R estimada: {det_R}")

        # El determinante debe ser aproximadamente 1
        self.assertTrue(np.isclose(det_R, 1.0, atol=1e-6), "Determinante de R no es 1")

if __name__ == '__main__':
    unittest.main()


    
