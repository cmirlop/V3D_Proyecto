import numpy as np
import unittest
#from Matriz_Esencial_E_et4 import calcular_matriz_essencial
from Matriz_esencial_apartado_4 import calcular_matriz_essencial



class TestFactorizacion(unittest.TestCase):




    def test_determinanteE(self):
        E = np.load('matriz_E.npy')
        det_E = np.linalg.det(E)
        self.assertTrue(np.isclose(det_E, 0.0, atol=1e-6), "Determinante de E no es 0")




    #COmprobamos que el determi:nante de la matriz de rotacion es 1
    def test_determinante_R(self):
        E = np.load('matriz_E.npy')
        # Suponiendo que tienes E cargada
        U, S, Vt = np.linalg.svd(E)
        # Recuperación de R (una de las posibles soluciones)
        W = np.array([[0, -1, 0],
                    [1,  0, 0],
                    [0,  0, 1]])
        R = U @ W @ Vt
        # Verifica que R sea válida
        det_R = np.linalg.det(R)
        print(f"Determinante de R estimada: {det_R}")
        # El determinante debe ser aproximadamente 1
        self.assertTrue(np.isclose(det_R, 1.0, atol=1e-6), "Determinante de R no es 1")

if __name__ == '__main__':
    unittest.main()


    
