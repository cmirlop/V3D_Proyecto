import time
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

from utils2 import save_point_cloud, render, median_blur

# Variables globales de diseño utilizadas para el ajuste de la obtencion del mapa de Disparidad
kernel_size = 15
max_disp = 64
subpixel = 'store_true'
kernel_half = int(kernel_size/2)
offset_adjust = int(255 / max_disp)

# --------------------------------------------------------------------------------------------------------

# Centra una imagen con un respectivo offset
def getWindowCentered(y, x, img, offset=0):
    y_start = y-kernel_half
    y_end = y+kernel_half
    x_start = x-kernel_half-offset+1
    x_end = x+kernel_half-offset+1
    return img[y_start:y_end,x_start:x_end]

# --------------------------------------------------------------------------------------------------------

# Comrpueba que el subpixel selecciona sea el mejor y lo cambia si es necesario, mediante la comprobacion de los pixeles que tiene alrededor
def getBestSubpixel(best_offset, errors):
    if 0 < best_offset < max_disp-1 and errors[best_offset-1] and errors[best_offset+1]:
        denom = errors[best_offset-1] + errors[best_offset+1] - 2*errors[best_offset]
        if denom != 0:
            subpixel_offset = (errors[best_offset-1] - errors[best_offset+1]) / (2*denom)
            return subpixel_offset
    return 0.0

# --------------------------------------------------------------------------------------------------------

# Genera el mapa de disparidad entre dos imagenes
def getDisparityMap(left, right):
    h, w = left.shape
    disp_map = np.zeros_like(left, dtype=np.float32)
    for y in range(kernel_half, h - kernel_half):      
        for x in range(max_disp, w - kernel_half):
            best_offset = None
            min_error = float("inf")
            errors = []
            for offset in range(max_disp):               
                W_left = getWindowCentered(y, x, left)
                W_right = getWindowCentered(y, x, right, offset)
                if W_left.shape != W_right.shape:
                    errors.append(None)
                    continue
                error = np.sum((W_left - W_right)**2)
                errors.append(np.float32(error))
                if error < min_error:
                    min_error = error
                    best_offset = offset
            if subpixel:
                best_offset += getBestSubpixel(best_offset, errors)
            disp_map[y, x] = best_offset * offset_adjust
    return disp_map

# --------------------------------------------------------------------------------------------------------

def main():
    # Carga las imagenes
    left = Image.open("vision3D/nube_3D/teddy-png-2/teddy/im2.png")
    right = Image.open("vision3D/nube_3D/teddy-png-2/teddy/im6.png")

    # Obtiene la disparidad a partir de la imagen izquierda y derecha
    start = time.time()
    disparity = getDisparityMap(np.array(left.convert('L')), np.array(right.convert('L')))
    end = time.time()

    # Filtra la imagen y obtiene los colores de los pixeles
    disparity = median_blur(disparity, 5)
    colors = np.array(left)
    print(f"Computation time: {end-start:.2f}s")

    # Crea la carpeta en caso de no existir para almacenar el archivo de la nube de puntos y el mapa de calor correspondiente en formato PNG
    if not os.path.exists("vision3D/nube_3D/output"):
        os.mkdir("vision3D/nube_3D/output")
    save_point_cloud(f"vision3D/nube_3D/output/BM2_python.ply", disparity, colors) # Guarda la nube de puntos en un archivo PLY
    plt.imsave(f"vision3D/nube_3D/output/BM2_python.png", disparity, cmap='jet') # Guarda el mapa de calor de la imagen en base a la nube de puntos

    # Muestra el resultado de la nube de puntos
    render("vision3D/nube_3D/output/BM2_python.ply")


if __name__ == "__main__":
    main()