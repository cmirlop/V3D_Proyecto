import cv2
from PIL import Image
import numpy as np

def calibracion(rutas_imagenes, tamano_tablero=(7, 5), tamano_cuadro=31, homografias=[]):
    V = []
    puntos_mundo = []
    for ruta in rutas_imagenes:
        imagen = Image.open(ruta).convert('RGB')

        '''if imagen.width != 450 or imagen.height != 375:
            nuevo_tamano = (450, 375)
            imagen = imagen.resize(nuevo_tamano)'''

        imagen = np.array(imagen)[:, :, ::-1]

        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Definir el tamaño del patrón del tablero de ajedrez (número de esquinas internas)
        # Por ejemplo, un tablero de 9x6 cuadros tiene 8x5 esquinas internas

        puntos_mundo = np.zeros((tamano_tablero[0] * tamano_tablero[1], 2), np.float32)
        puntos_mundo[:, :] = np.mgrid[0:tamano_tablero[0], 0:tamano_tablero[1]].T.reshape(-1, 2)
        puntos_mundo *= tamano_cuadro

        # Encontrar las esquinas del tablero
        encontrado, esquinas = cv2.findChessboardCorners(gris, tamano_tablero, None)

        esquinas_lista = esquinas.reshape(-1, 2).tolist()
        esquinas_lista = np.array(esquinas_lista)

        # Si se encuentran las esquinas, refinarlas y mostrarlas
        if encontrado:
            # Refinar las ubicaciones de las esquinas
            criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            esquinas = cv2.cornerSubPix(gris, esquinas, (11,11), (-1,-1), criterios)

            # Dibujar las esquinas en la imagen
            #cv2.drawChessboardCorners(imagen, tamano_tablero, esquinas, encontrado)

            # Mostrar la imagen
            #cv2.imshow('Esquinas del tablero', imagen)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
        else:
            print("No se encontraron las esquinas del tablero.")
            
        h = calcular_homografia_dlt(puntos_mundo, esquinas_lista)
        homografias.append(h)
        mean_error, individual_errors = reprojection_error(h, puntos_mundo, esquinas_lista)
        #print("Error de reproyección promedio:", mean_error)
        v = matriz_v(h)
        V.append(v)

        #homografias.append(H)
    V = np.array(V)
    V = V.reshape(-1, 6)
    B = matriz_b(V)
    K = matriz_k(B)
    '''K = np.linalg.cholesky(B)
    K = np.linalg.inv(K).T
    K = K / K[2, 2]'''
    return K

def reprojection_error(H, src_pts, dst_pts):
    """
    Calcula el error de reproyección dado H y listas de puntos correspondientes.
    - H: matriz 3x3
    - src_pts: Nx2 puntos originales (x, y)
    - dst_pts: Nx2 puntos destino esperados (x', y')
    """

    src_homog = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))  # Nx3
    projected = (H @ src_homog.T).T  # Nx3

    # Normalizar coordenadas homogéneas
    projected /= projected[:, [2]]  # divide x, y por w

    # Calcular distancias euclidianas
    error = np.linalg.norm(dst_pts - projected[:, :2], axis=1)
    mean_error = np.mean(error)

    return mean_error, error

def calcular_homografia_dlt(puntos_origen, puntos_destino):
    # Calcula la matriz de homografía usando el algoritmo DLT a partir de 4 puntos.
    A = []
    for i in range(len(puntos_origen)):
        (x,y) = puntos_origen[i]
        (xd,yd) = puntos_destino[i]
        a = np.array([[-x,-y, -1, 0, 0, 0, x*xd, y*xd, xd],
                    [0,0, 0, -x, -y, -1, x*yd, y*yd, yd]])
        A.append(a)
    A = np.vstack(A)

    u, d, vT = np.linalg.svd(A)

    H = vT[-1].reshape(3, 3)
    # IMPLEMENTA EL ALGORITMO DLT
    return H / H[2, 2]

def v_ij(H, i, j):
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[1, i] * H[1, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])

def matriz_v(matriz):
    v_12 = v_ij(matriz, 0, 1)
    v_11 = v_ij(matriz, 0, 0)
    v_22 = v_ij(matriz, 1, 1)

    v = np.vstack((v_12, v_11-v_22))
    
    v = np.array(v)
    return v

def matriz_b(V):
    u, d, vT = np.linalg.svd(V)
    b = vT[-1]
    B = b = np.array([[b[0],b[1],b[2]],
                  [b[1],b[3],b[4]],
                  [b[2],b[4],b[5]]])
    return B / B[2, 2]

def matriz_k(B):
    b11, b12, b13 = B[0, 0], B[0, 1], B[0, 2]
    b22, b23 = B[1, 1], B[1, 2]
    b33 = B[2, 2]

    delta = b11 * b22 - b12**2
    if delta <= 0:
        raise ValueError("DELTA: La matriz B no es válida para la calibración.")
    
    k13 = (b12 * b23 - b13 * b22) / delta
    k23 = (b12 * b13 - b11 * b23) / delta

    lam = k13 * b13 + k23 * b23 + b33
    if lam <= 0:
        raise ValueError("LAMBDA: La matriz B no es válida para la calibración.")
    
    k11 = np.sqrt(lam / b11)
    k22 = np.sqrt((lam * b11) / delta)
    k12 = - (b12 * k11**2 *k22) / lam
    k33 = 1.0

    K = np.array([
        [k11, k12, k13],
        [0.0, k22, k23],
        [0.0, 0.0, k33]
    ])   
    np.save('output/matriz_K.npy', K)
    return K

def matriz_p(K, h):
    Rt_aprox = np.linalg.inv(K) @ h
    r1 = Rt_aprox[:, 0]
    r2 = Rt_aprox[:, 1]
    t = Rt_aprox[:, 2]

    norm = np.linalg.norm(r1)
    r1 = r1 / norm
    r2 = r2 / norm
    t = t / norm
    print("t:", t)

    r3 = np.cross(r1, r2)

    R = np.column_stack((r1, r2, r3))

    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    print("Determinante de R:", np.linalg.det(R))
    print("R:", R)

    P = K @ np.column_stack((R, t))
    np.save('output/matriz_P.npy', P)
    return P