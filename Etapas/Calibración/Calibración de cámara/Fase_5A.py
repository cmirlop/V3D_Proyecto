import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Cargar la matriz fundamental F desde el archivo exportado
def cargar_matriz_E():
    return np.load('matriz_E.npy')

# Cargar la matriz fundamental F desde el archivo exportado
def cargar_matriz_F():
    return np.load('matriz_F.npy')

def cargar_matriz_K():
    return np.load('matriz_K.npy')

# Detectar puntos clave y descriptores con SIFT
def apl_shift(img_left, img_right):
    sift = cv2.SIFT_create()
    kp_left, des_left = sift.detectAndCompute(img_left, None)
    kp_right, des_right = sift.detectAndCompute(img_right, None)
    return kp_left,des_left,kp_right,des_right

# Emparejar puntos clave usando BFMatcher
def apl_matcher(des_left,des_right):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_left, des_right, k=2)
    return matches

# Aplicar el filtro de razón de Lowe
def filtro_lowe(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

# Obtener puntos clave buenos
def obtener_puntos_buenos(kp_left,kp_right,good_matches):
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])
    return pts_left,pts_right


#Convertir en homogeneas
def conv_homogeneas(pts_left, pts_right):
    pts_left = np.hstack([pts_left, np.ones((pts_left.shape[0], 1))])
    pts_right = np.hstack([pts_right, np.ones((pts_right.shape[0], 1))])
    return pts_left, pts_right


#Función que lee las imagenes y las convierte a escala de grises
def cargar_imagenes():
    # Cargar las imágenes y convertirlas a escala de grises
    image_left = cv2.imread('data/left.png')
    image_right = cv2.imread('data/right.png') 

    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    return image_left_gray, image_right_gray, image_left, image_right

def rectificacion_Esteroscipica_calibrada(E,y1,y2):

    #Obtenemos SVD
    U,S,Vt = np.linalg.svd(E)

    #Creamos la matriz W
    W = np.array([[0, 1, 0],
              [-1, 0, 0],
              [0, 0, 1]])

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
        u1, v1, _ = y1
        u2, v2, _ = y2

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
        x = x / x[2]  # Normalizar homogéneo

        #2.- Compute the same point in the camera 
        # centered coordinate system of the second camera:
        print(pose[0])
        print(x)
        print(pose[1])
        x_prim = pose[0] @ x[:3].T + pose[1]
        #3.- Return (R,t) si x3 > y x3'>0
        if x[2] > 0 and x_prim[2] > 0 :
            return pose



#--- Pasos ---#

#1.-Leemos imagenes y las convertimos de color a gris
img_left, img_right, imgI, imgD= cargar_imagenes()

#2.- Detectamos puntos clave y descriptores
kp_left,des_left,kp_right,des_right = apl_shift(img_left,img_right)

#3.- Emparejamiento de puntos clave
matches = apl_matcher(des_left,des_right)

#4.- Aplicamos filtro de Lowe
good_matches = filtro_lowe(matches)

#5.- Obtener puntos clave buenos
pts_left, pts_right = obtener_puntos_buenos(kp_left,kp_right,good_matches)

#6.- Cargar la matriz E en el código
E = cargar_matriz_E()
#F = cargar_matriz_F()

#7.- Convertir en coordenadas homogeneas
pts_left,pts_right=conv_homogeneas(pts_left, pts_right)

i = np.random.randint(len(pts_left))

#8.- Aplicar el cóigo de Rectificación 
Rectificacion = rectificacion_Esteroscipica_calibrada(E,pts_left[i],pts_right[i])

print("Rectificacion")
print(Rectificacion)


K = cargar_matriz_K()
D = np.diag(np.sign(np.diag(K)))
K = K @ D
print(img_left.shape)
print("K")
print(K)
print(np.linalg.det(K))


# Paso 1: eje base r1 normalizado
r1 = Rectificacion[1] / np.linalg.norm(Rectificacion[1])

# Paso 2: vector vertical del mundo
ez = np.array([0, 0, 1])

# Paso 3: r2 perpendicular a r1 y ez
r2 = np.cross(ez, r1)
r2 = r2 / np.linalg.norm(r2)

# Paso 4: r3 perpendicular a r1 y r2
r3 = np.cross(r1, r2)

# Paso 5: matriz de rotación común (filas r1, r2, r3)
R_rect = np.vstack([r1, r2, r3])

# Paso 6: rotaciones para cada cámara
R1 = R_rect
R2 = R_rect @ Rectificacion[0]

# Paso 7: homografías
K_inv = np.linalg.inv(K)
Hl = K @ R1.T @ K_inv
Hr = K @ R2.T @ K_inv

print(Hl)


print(Hr)

Hl = Hl / Hl[2,2]
Hr = Hr / Hr[2,2]




img_left_rect = cv2.warpPerspective(imgI, Hl, (img_left.shape[1], img_left.shape[0]))
img_right_rect = cv2.warpPerspective(imgD, Hr, (img_right.shape[1], img_right.shape[0]))

# Crear una imagen combinada
combined_image = np.hstack((img_left_rect, img_right_rect))

# Visualización mejorada
plt.figure(figsize=(15, 5))
plt.imshow(combined_image, cmap='gray')
plt.show()

'''
dist1 = np.zeros(5)  # o tu vector de distorsión real
dist2 = np.zeros(5)

R = Rectificacion[0]  # Rotación relativa (3x3)
T = Rectificacion[1] # Traslación relativa (3x1)

# Tamaño imagen (ancho, alto)
img_size = (img_left.shape[1], img_left.shape[0])

# Rectificación estéreo
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    K, dist1,
    K, dist2,
    img_size, R, T,
    flags=cv2.CALIB_ZERO_DISPARITY,
    alpha=0
)

# Calcular homografías para rectificar imágenes
Hl = K @ R1 @ np.linalg.inv(K)
Hr = K @ R2 @ np.linalg.inv(K)

print("Homografía para imagen izquierda:\n", Hl)
print("Homografía para imagen derecha:\n", Hr)

img_left_rect = cv2.warpPerspective(img_left, Hl, (img_left.shape[1], img_left.shape[0]))
img_right_rect = cv2.warpPerspective(img_right, Hr, (img_right.shape[1], img_right.shape[0]))

# Crear una imagen combinada
combined_image = np.hstack((img_left_rect, img_right_rect))

# Visualización mejorada
plt.figure(figsize=(15, 5))
plt.imshow(combined_image, cmap='gray')
plt.show()
'''