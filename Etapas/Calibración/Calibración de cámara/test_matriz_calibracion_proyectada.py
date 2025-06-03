import unittest
import numpy as np
import os
from Matriz_calibración_proyectada_P_et1 import main_p

class TestMatrizCalibracionProyectada(unittest.TestCase):
    def setUp(self):
        # Usa imágenes de prueba del workspace
        self.image_paths = [
            'data/my_frame-00.jpg',
            'data/my_frame-01.jpg',
            'data/my_frame-02.jpg',
            'data/my_frame-03.jpg',
            'data/my_frame-04.jpg',
            'data/my_frame-05.jpg',
            'data/my_frame-06.jpg',
            'data/my_frame-07.jpg',
        ]
        # Elimina el archivo de salida si existe
        if os.path.exists('matriz_P.npy'):
            os.remove('matriz_P.npy')

    def test_main_p_runs_and_saves_P(self):
        P = main_p(self.image_paths)
        # Comprueba que la matriz P tiene la forma correcta
        self.assertEqual(P.shape, (3, 4))
        # Comprueba que el archivo se ha guardado
        self.assertTrue(os.path.exists('matriz_P.npy'))
        P_loaded = np.load('matriz_P.npy')
        np.testing.assert_allclose(P, P_loaded, rtol=1e-6)

    def test_P_is_finite(self):
        P = main_p(self.image_paths)
        self.assertTrue(np.all(np.isfinite(P)))

    def test_P_last_element_is_one(self):
        P = main_p(self.image_paths)
        self.assertAlmostEqual(P[-1, -1], 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
