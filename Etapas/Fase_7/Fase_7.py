import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import open3d as o3d
import cv2 as cv

'''Para el desarrollo del código correspondiente de la generación de la nube de puntos 3D. 
Al principio del código se han declarado unas variables globales las cuales se utilizan en 
la implementación del algoritmo Block Matching. Pero para facilitar su ajuste se han dejado 
al principio. Estas variables son:'''
block_size = 15 # Es el tamaño del bloque y corresponde al área de pixel que se van a analizar. Contra menor se a el valor se obtiene mas detalle, pero a su vez también se obtiene mas ruido.
max_disp = 32 # Es la disparidad máxima y sirve para indicar el rango de comparación con los pixeles de alrededor del bloque. 
activate_subpixel = True # Permite activar la interpolación subpixel
block_half = int(block_size/2) # Es un atajo para centrar obtener el centro del bloque.

# -------------------------------------------------- ------------------------------------------------------

# Obtiene la región de interés o bloque que se desea analizar. Permitiendo introducir un 
# offset(desplazamiento) sobre el eje X para adecuar el bloque en una comparación horizontal.
def getROI(y, x, img, desplazamiento=0): 
    y_start, y_end = y - block_half, y +block_half
    x_start, x_end = x - block_half - desplazamiento + 1, x + block_half - desplazamiento + 1
    return img[y_start:y_end, x_start:x_end]

# -------------------------------------------------- ------------------------------------------------------


#Permite seleccionar la funcion de coste que se quiera entre SAD, SSD y NCC
def fdc(block_prev, block_next, mode=1):
    if mode == 0: # SAD (Sum of Absolute Differences)
        error = np.sum(np.abs(block_prev - block_next), dtype=np.float32)

    elif mode == 1: # SSD (Sum of Squared Differences)
        error = np.sum((block_prev - block_next)**2, dtype=np.float32)

    elif mode == 2: # NCC (Normalized Cross-Correlation)
        numerator = np.sum(block_prev * block_next)
        denominator = np.sqrt(np.sum(block_prev ** 2) * np.sum(block_next ** 2))
        if denominator != 0:
            error = numerator / denominator
        else:
            error  = 0.0

    else:
        return "Opcion no valida"
    
    return error

    pass
# -------------------------------------------------- ------------------------------------------------------


# Comrpueba que el subpixel selecciona sea el mejor y lo cambia si es necesario, mediante la comprobacion de los pixeles que tiene alrededor.
# Generando puntos entre las capas del Block Matching
def getBestSubpixel(best_offset, errors):
    if 0 < best_offset < max_disp-1 and errors[best_offset-1] and errors[best_offset+1]:
        numerador = errors[best_offset-1] - errors[best_offset+1]
        denominator = 2 * errors[best_offset-1] - 4 * errors[best_offset] + 2 * errors[best_offset+1] 
        if denominator != 0:
            subpixel_offset = (numerador / denominator)
            return subpixel_offset
    return 0.0

# -------------------------------------------------- ------------------------------------------------------

# Elige un pixel aleatorio de la imagen izquierda y lo compara con la fila 
# correspondiente en la imagen derecha. Aplicando las funciones de coste 
# SAD, SSD y NCC. Por ultimo genera una grafica de comparacion del error obtenido
# con cada funcion y la guarda en la carpeta output
def compare_random_pixel(left, right, saveRoute):
    h, w = left.shape

    random_x = np.random.randint(0, w)
    random_y = np.random.randint(0, h)
    random_x = 290
    random_y = int(h/2)

    errorsSAD=[]
    errorsSSD=[]
    errorsNCC=[]

    block_left = getROI(random_y, random_x, left)
    cv.imwrite(saveRoute+"block_left.png", block_left)

    bloques = []

    for x in range(max_disp):
        block_right = getROI(random_y, random_x, right, x)
        bloques.append(block_right)

        error = fdc(block_left, block_right, 1) # Funcion de coste(0-SAD,1-SSD,2-NCC)
        errorsSAD.append(error)
        error = fdc(block_left, block_right, 0)
        errorsSSD.append(error)
        error = fdc(block_left, block_right, 2)
        errorsNCC.append(error)

    concat_bloques = cv.hconcat(bloques)
    cv.imwrite(saveRoute+"concat_bloques.png", concat_bloques)

    # Crea la figura y los ejes
    fig = plt.figure(dpi=100)
    plt.plot(np.arange(len(errorsSAD)), errorsSAD, label="SAD", color="b")
    plt.plot(np.arange(len(errorsSSD)), errorsSSD, label="SSD", color="r")
    plt.plot(np.arange(len(errorsNCC)), errorsNCC, label="NCC", color="g")

    # Etiquetas y título
    plt.xlabel("X")
    plt.ylabel("Error")
    plt.title(f"Gráfica de la función de coste en el pixel {random_x},{random_y}")
    plt.legend()
    plt.grid(True)

    # Muestra la gráfica
    plt.show()

    # Guarda la grafica
    fig.savefig(saveRoute+"SADvsSSDvsNCC.png", dpi=300)

# -------------------------------------------------- ------------------------------------------------------


''' Genera el mapa de disparidad entre dos imagenes, de la imagen izquierda obtiene sus dimensiones 
y se utilizan para generar el mapa vació de disparidad. Luego se recorren todas las posiciones de la
 imagen, empezando en el eje Y. Pero no se empieza por 0, sino por la mitad de distancia del bloque, 
 debido que en caso de no hacer esto se produce un error al utilizarlo mas adelante. Después se 
 recorre el eje X por la posición correspondiente al valor de disparidad máxima, para que no se 
 salga a la hora de realizar la comparación.

Una vez posicionados se genera el bloque de la imagen de origen(imagen izquierda) y se inicializan 
los parámetros correspondientes a ese bloque(mejor error, mejor desplazamiento y lista de los 
mejores errores). A continuación se compara el bloque origen con bloques que se generan sobre la 
imagen destino(imagen derecha). Estos bloques se van desplazando en un rango de 0 a max\_disp-1, es 
decir, que se va moviendo el bloque y nos quedamos con el que tiene menor error.

Tras haber obtenido el menor error, se llama a la función subpixel comentada anteriormente. Por ultimo 
se almacena el valor en la posición correspondiente a los pixeles de la imagen.
'''
def getDisparityMap(left, right):

    h, w = left.shape
    disp_map = np.zeros((h,w), dtype=np.float32)

    for y in range(block_half, h):
        for x in range(max_disp, w):   
            block_prev = getROI(y, x, left)
            best_error = float('inf')
            best_d = None
            errors = []

            for dx in range(max_disp): # Compara el pixel del bloque con los pixeles que tiene alrededor en el eje X
                block_next = getROI(y, x, right, dx)

                if block_prev.shape != block_next.shape:
                    errors.append(None)
                    continue

                error = fdc(block_prev, block_next, 1) # Funcion de coste(0-SAD,1-SSD,2-NCC)
                errors.append(error)

                if error < best_error:
                    best_error = error
                    best_d = dx
                    
            if activate_subpixel and best_d != None:
                best_d += getBestSubpixel(best_d, errors)

            disp_map[y, x] = best_d

    return disp_map

# -------------------------------------------------- ------------------------------------------------------

# Recorre todo el mapa de disparidad y va obteniendo la profundidad correspondiente a cada pixel.
def reproject_image_to_3D(disparity, T_1):
    height, width = disparity.shape
    points_3D = np.zeros((height, width, 3), dtype=np.float32)
    msg = "\rLoading ."
    sys.stdout.write(f"{msg}")

    for y in range(height):
        for x in range(width):

            if disparity[y, x] == 0: # Evita division entre 0 y establece que el punto esta al frente
                points_3D[y, x] = 0
                continue

            d = disparity[y, x]
            vec = np.array([x, y, d, 1], dtype=np.float32)  # Pasa a coordenadas homogeneas
            point = T_1 @ vec  
            points_3D[y, x] = point[:3] / point[3]  # Normaliza con W, es decir, el vector entre la ultima componente para dejarla a 1 

        # Escribe Loading ... con algo de movimiento en los 3 puntos
        if len(msg) <= 11:
            msg += "."
        else:
            msg = "\rLoading ."
        sys.stdout.write(f"{msg}")
        sys.stdout.flush()
        
    return points_3D

# -------------------------------------------------- ------------------------------------------------------

# Esta función se encarga simplemente de la generación de una ventana con un visor 3D de la nube de puntos 
# que se ha generado con la función anterior y ha sido guardada en el archivo PLY.
def render(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

# -------------------------------------------------- ------------------------------------------------------

# Filtra la imagen en escala de grises, reduciendo así el ruido de la imagen mediante el uso del tamaño del bloque(ksize). 
def median_blur(image, ksize):
    if ksize % 2 == 0:
        raise ValueError("El tamanyo del bloque debe ser un numero impar.")

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

# -------------------------------------------------- ------------------------------------------------------


# Guarda la nube de puntos 3D con los colores en un archivo PLY. 
# 
# Utilizando la matriz de calibración(matriz K), el mapa de disparidad y el valor 
# RGB de cada posición, se guardan en un archivo PLY de forma conjunta. Para que 
# luego se puedan utilizar en el visor 3D.
def save_point_cloud(filename, disparity, colors):
    K = np.load('matriz_K.npy')
    
    cx = K[0,2]
    cx_p = -cx
    cy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]
    Tx = K[1,2]

    T_1 = np.array([[1/fx, 0, 0, -cx/fx],#-1.97
                [0, 1/fy, 0, -cy/fy],#1.01
                [0, 0, 0, 1],  # distancia focal -3.5
                [0, 0, 1/(fx*Tx), (cx_p-cx)/(fx*Tx)]]) 
    
    points_3d = reproject_image_to_3D(disparity, T_1)
    mask = disparity > 0  # Elimina los puntos en los que la disparidad en 0 o menos
    
    points = points_3d[mask]
    colors = colors[mask]
    
    points = np.hstack([points, colors])
    
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
        
# -------------------------------------------------- ------------------------------------------------------
