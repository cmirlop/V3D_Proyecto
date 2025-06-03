import unittest
import numpy as np
import cv2
from Matriz_Fundamental_F_Et3 import calcular_fundamental, apl_sift, apl_matcher, filtro_lowe, obtener_puntos_buenos

class TestMatrizFundamental(unittest.TestCase):

    def test_rango_matriz_f(self):
        F = np.load('matriz_F.npy')
        rank = np.linalg.matrix_rank(F)
        self.assertEqual(rank, 2, "La matriz fundamental debe ser de rango 2")

    
if __name__ == '__main__':
    unittest.main()
