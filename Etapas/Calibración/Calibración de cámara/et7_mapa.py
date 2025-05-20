
import numpy as np

def mapa_puntos(img_left,img_right, blockSize):
    heigh_l, weith_r = img_left.shape
    disp_map = np.zeros_like(img_left, dtype=np.float32)
    for x in heigh_l:
        for y in weith_r:
            pass