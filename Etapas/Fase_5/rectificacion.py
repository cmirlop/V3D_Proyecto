import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image, ImageDraw

def epipolo(F):
    _, _, Vt = np.linalg.svd(F.T)
    e = Vt[-1]
    return e / e[2]

def matriz_M(F):
    # Descomponemos la matriz F
    u, s, vt = np.linalg.svd(F)

    # Calculamos el valor de lambda
    lam = (s[0] + s[1]) / 2

    # Modificamos los valores singulares
    S = np.array([[0, s[1], 0],
                  [-s[0], 0, 0],
                  [0, 0, lam]])

    # Recomponemos para obtener la matriz M
    M = u @ S @ vt
    return M

def homografias_rectificadas(puntos_izq, puntos_dcha, punto, F):
    puntos_izq_transformados = np.zeros((puntos_izq.shape[0], 3))
    puntos_dcha_transformados = np.zeros((puntos_dcha.shape[0], 3))

    # Obtenemos el epipolo izquierdo
    e_izq = epipolo(F)
    print("Valor comprobación = ", F@e_izq)

    # Obtenemos la matriz M
    M = matriz_M(F)

    # Obtenemos la matriz de transformación T
    T_trans = np.array([[1, 0, -punto[0]],
                   [0, 1, -punto[1]],
                   [0, 0, 1]])
    
    # Sacamos el epipolo trasladado
    e_izq_trasladado = T_trans @ e_izq

    # Con el epipolo trasladado sacamos la matriz de rotacion
    alpha = np.arctan2(e_izq_trasladado[1], e_izq_trasladado[0])
    if alpha > np.pi / 2:
        alpha += np.pi
    T_rot = np.array([[np.cos(alpha), np.sin(alpha), 0],
                   [-np.sin(alpha), np.cos(alpha), 0],
                   [0, 0, 1]])
    
    # Sacamos el epipolo rectificado ahora
    e_izq_rectificado = T_rot @ e_izq_trasladado

    # Ahora sacmos la homografia al infinito de la imagen izquierda
    Hinf = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [-1/e_izq_rectificado[0], 0, 1]])
    
    # Sacamos la homografia de la imagen izquierda
    Hl = Hinf @ T_rot @ T_trans
    Hl = Hl / Hl[2, 2]

    # Transformamos las imagenes
    puntos_izq = np.hstack((puntos_izq, np.ones((puntos_izq.shape[0], 1))))
    puntos_dcha = np.hstack((puntos_dcha, np.ones((puntos_dcha.shape[0], 1))))

    for i in range(puntos_izq.shape[0]):
        p_izq_hom = Hl @ puntos_izq[i]
        p_izq_hom /= p_izq_hom[2]
        puntos_izq_transformados[i] = p_izq_hom

        p_dcha_hom = Hl @ M @ puntos_izq[i]
        p_dcha_hom /= p_dcha_hom[2]
        puntos_dcha_transformados[i] = p_dcha_hom

    # Montar la matriz Y
    Yl = puntos_izq_transformados.T
    Yr = puntos_dcha_transformados.T
    ur = Yr[0]
    ur = ur.reshape(1, -1)

    # Montamos las partes para resolver la ecuacuion
    ul = Yl[0]
    A = Yr @ Yr.T
    b = Yr @ ul.T

    a = np.linalg.solve(A, b)

    # Calculamos la matriz A
    A = np.array([[a[0], a[1], a[2]],
                  [0, 1, 0],
                  [0, 0, 1]])
    
    # Sacamos la homografia de la imagen derecha
    Hr = A @ Hl @ M
    Hr = Hr / Hr[2, 2]
    return Hl, Hr

def aplicar_homografia(imagen, H):
    imagen = np.array(imagen)
    h, w = imagen.shape[:2]

    # Paso 1: Calcular el bounding box de la imagen transformada
    esquinas = np.array([
        [0, 0, 1],
        [w, 0, 1],
        [0, h, 1],
        [w, h, 1]
    ]).T  # 3x4

    esquinas_transformadas = H @ esquinas
    esquinas_transformadas /= esquinas_transformadas[2]

    min_x = np.floor(np.min(esquinas_transformadas[0])).astype(int)
    max_x = np.ceil(np.max(esquinas_transformadas[0])).astype(int)
    min_y = np.floor(np.min(esquinas_transformadas[1])).astype(int)
    max_y = np.ceil(np.max(esquinas_transformadas[1])).astype(int)

    new_w = max_x - min_x
    new_h = max_y - min_y

    # Paso 2: Compensar la traslación con una matriz de desplazamiento
    T = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])
    H_corr = T @ H  # Homografía corregida

    # Paso 3: Crear malla de coordenadas en la imagen destino
    x_coords, y_coords = np.meshgrid(np.arange(new_w), np.arange(new_h))
    homog_coords = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords.ravel())])

    H_inv = np.linalg.inv(H_corr)
    coords_fuente = H_inv @ homog_coords
    coords_fuente /= coords_fuente[2]

    x_src = coords_fuente[0].reshape(new_h, new_w)
    y_src = coords_fuente[1].reshape(new_h, new_w)

    # Paso 4: Interpolación por canales
    imagen_rectificada = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    for i in range(3):  # R, G, B
        imagen_rectificada[:, :, i] = map_coordinates(imagen[:, :, i], [y_src, x_src], order=1, mode='constant', cval=0)

    return Image.fromarray(imagen_rectificada)

'''def aplicar_homografia(imagen, H):
    imagen = np.array(imagen)
    H = H / H[2, 2]
    h_salida, w_salida = imagen.shape[:2]

    # Crear una malla de coordenadas
    x_coords, y_coords = np.meshgrid(np.arange(w_salida), np.arange(h_salida))

    coordenadas_homogeneas = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones_like(x_coords.ravel())])

    H_inv = np.linalg.inv(H)
    coordenadas_iniciales = H_inv @ coordenadas_homogeneas
    coordenadas_iniciales /= coordenadas_iniciales[2, :]

    x_iniciales = coordenadas_iniciales[0, :].reshape(h_salida, w_salida)
    y_iniciales = coordenadas_iniciales[1, :].reshape(h_salida, w_salida)

    imagen_rectificada = np.zeros_like(imagen, dtype=np.uint8)
    for i in range(3):
        imagen_rectificada[:, :, i] = map_coordinates(imagen[:, :, i], [y_iniciales, x_iniciales], order=1, mode='constant', cval=0)

    return Image.fromarray(imagen_rectificada)'''

def dibujar_rectificaciones(imagen_izq, imagen_dcha):
    h = min(imagen_izq.height, imagen_dcha.height)
    w = imagen_izq.width + imagen_dcha.width

    imagen = Image.new("RGB", (w, h))
    imagen.paste(imagen_izq, (0,0))
    imagen.paste(imagen_dcha, (imagen_izq.width, 0))

    dibujo = ImageDraw.Draw(imagen)
    paso = h // (10 + 1)

    for i in range(1, 10 + 1):
        y = i * paso
        dibujo.line([(0, y), (w, y)], fill=(255,255,255), width=1)

    return imagen