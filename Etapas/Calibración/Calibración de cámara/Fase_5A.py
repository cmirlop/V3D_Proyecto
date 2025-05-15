import numpy as np
import cv2

# Cargar la matriz fundamental F desde el archivo exportado
def cargar_matriz_E():
    return np.load('matriz_E.npy')

#Función que lee las imagenes y las convierte a escala de grises
def cargar_imagenes():
    # Cargar las imágenes y convertirlas a escala de grises
    image_left = cv2.imread('data/left.png') 

    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    return image_left_gray

def rectificacion_Esteroscipica_calibrada(E,y1,y2):

    #Obtenemos SVD
    U,S,Vt = np.linalg.svd(E)

    #Creamos la matriz W
    W = np.array([0,1,0],[-1,0,0],[0,0,1])

    #Poses camara
    t_hat = Vt[:, 2] #Ultima columna 
    P1 = [Vt.T @ W @ U.T, t_hat]
    P2 = [Vt.T @ W.T @ U.T, t_hat]
    P3 = [Vt.T @ W @ U.T, t_hat]
    P4 = [Vt.T @ W.T @ U.T, t_hat]
    poses_camara = [P1, P2, P3, P4]
    for pose in poses_camara:
        #1.- Triagulacion 3D
        #Matriz de proyeccion de la camara principal (Identidad y cero)
        P1 = np.hstack((np.eye(3), np.zeros((3,1))))
        #Matriz de la segunda camara (R|T)
        P2 = np.hstack((pose[0], pose[1].reshape(3,1)))

        # Extraemos u y v de los puntos proporcionados
        u1, v1 = y1
        u2, v2 = y2

        # Construir la matriz A (4x4)
        A = np.array([
            u1 * P1[2,:] - P1[0,:],
            v1 * P1[2,:] - P1[1,:],
            u2 * P2[2,:] - P2[0,:],
            v2 * P2[2,:] - P2[1,:]
        ])

        # Resolver A X = 0 usando SVD
        _, _, Vt = np.linalg.svd(A)
        x = Vt[-1]
        x = x / x[3]  # Normalizar homogéneo

        #2.- Compute the same point in the camera 
        # centered coordinate system of the second camera:
        x_prim = pose[0] @ x + pose[1]
        #3.- Return (R,t) si x3 > y x3'>0
        if x[2] > 0 and x_prim[2] > 0 :
            return pose


#Leemos imagenes y las convertimos de color a gris
img_left= cargar_imagenes()
# Detectar puntos clave y descriptores con SIFT
sift = cv2.SIFT_create()
kp_left, des_left = sift.detectAndCompute(img_left, None)

# Emparejar puntos clave usando BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_left, k=2)

# Aplicar el filtro de razón de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Obtener puntos clave buenos
pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])

E = cargar_matriz_E()
Rectificacion = rectificacion_Esteroscipica_calibrada(E,pts_left[20],pts_left[30])