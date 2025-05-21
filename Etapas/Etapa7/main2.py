import time
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import open3d as o3d
#from utils2 import save_point_cloud, render, median_blur

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

def reproject_image_to_3D(disparity, Q):
    height, width = disparity.shape
    points_3D = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            d = disparity[y, x] if disparity[y, x] > 0 else 0.0001  # Evita la division por 0
            vec = np.array([x, y, d, 1], dtype=np.float32)  # Pasa a coordenadas homogeneas
            point = Q @ vec  
            points_3D[y, x] = point[:3] / point[3]  # Normaliza con W

    return points_3D

# --------------------------------------------------------------------------------------------------------

def render(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

# --------------------------------------------------------------------------------------------------------

def median_blur(image, ksize):
    if ksize % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser un número impar.")

    height, width = image.shape
    padded_image = np.pad(image, ksize // 2, mode='reflect')
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Extrae la vecindad del píxel
            neighborhood = padded_image[i:i+ksize, j:j+ksize]
            # Calcula la mediana y la asigna al píxel correspondiente
            output[i, j] = np.median(neighborhood)

    return output

# --------------------------------------------------------------------------------------------------------

# Guarda la nube de puntos 3D con los colores en un archivo PLY
def save_point_cloud(filename, disparity, colors):
    cx=-1.97 * 100
    cx_p=-cx
    cy=1.01 * 100
    f=-3.5 * 100
    Tx=1 * 0.01


    '''
    Q = np.array([[1, 0, 0, -disparity.shape[1]/2],
                [0, -1, 0, disparity.shape[0]/2],
                [0, 0, 0, -0.8*disparity.shape[1]],  # distancia focal; adjust based on calibration
                [0, 0, 1/0.05, 0]])  # linea de base (adjust based on your setup)
    '''
    Q = np.array([[1, 0, 0, -cx],#-1.97
                [0, 1, 0, -cy],#1.01
                [0, 0, 0, f],  # distancia focal -3.5
                [0, 0, -1/Tx, (cx-cx_p)/Tx]])  # linea de base (-1.01 - (-1.01))/1
    ''''''
    #points_3d = cv.reprojectImageTo3D(disparity, Q)
    points_3d = reproject_image_to_3D(disparity, Q)
    mask = disparity > 0  # Elimina los puntos en los que la disparidad en 0 o menos
    #print(f"MMMMMMMMMMMMMMMM{disparity.shape[0]}, {disparity.shape[1]}")
    points = points_3d[mask]
    colors = colors[mask]
    #print(f"------------{points}, {colors}")
    points = np.hstack([points, colors])
    #print(f"************{points}")
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%f %f %f %d %d %d")
        
# --------------------------------------------------------------------------------------------------------

#import cv2 as cv
def main():
    # Carga las imagenes
    left = Image.open("vision3D/nube_3D/teddy-png-2/teddy/left.png")
    right = Image.open("vision3D/nube_3D/teddy-png-2/teddy/right.png")

    if left.width > 800:
        new_size = (left.width // 4, left.height // 4)
        left = left.resize(new_size)
        right = right.resize(new_size)

    # Obtiene la disparidad a partir de la imagen izquierda y derecha
    start = time.time()
    disparity = getDisparityMap(np.array(left.convert('L')), np.array(right.convert('L')))
    end = time.time()

    # Filtra la imagen y obtiene los colores de los pixeles de la imagen izquierda
    disparity = median_blur(disparity, 5)
    colors = np.array(left)

    #colors = [color[:, :-1] for color in colors]

    #print(f"PPPPPPPPPPPPP{colors}")
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