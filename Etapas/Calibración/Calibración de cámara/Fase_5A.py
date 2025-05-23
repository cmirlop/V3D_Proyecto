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

def cargar_matriz_R():
    return np.load('matriz_R_et2.npy')

def cargar_matriz_t():
    return np.load('matriz_t_et2.npy')

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
    image_left = cv2.imread('data/left2.png')
    image_right = cv2.imread('data/right2.png') 

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
        B = np.linalg.norm(pose[1])
        #1.- Triagulacion 3D
        #Matriz de proyeccion de la camara principal (Identidad y cero)
        #P1 = np.hstack((np.array([[-3.5,0,1.97],[0,-3.5,1.01],[0,0,1]]), np.zeros((3,1))))
        P1 = np.hstack((np.eye(3,3), np.zeros((3,1))))
        #Matriz de la segunda camara (R|T)
        P2 = np.hstack((pose[0], pose[1].reshape(3,1)))
        #P2 = np.hstack(((np.array([[-3.5,0,1.97],[0,-3.5,1.01],[0,0,1]]),np.array([[B*-3.5,0,0]]).T)))

        Q = np.array(([[1,0,0, -1.97],[0,1,0,-1.01],[0,0,0,-3.5],[0,0,-1/B,(-3.5+3.5)/B]]))
        print(P1)
        print(P2)
        print(Q)
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
        x = x / x[-1]  # Normalizar homogéneo

        #2.- Compute the same point in the camera 
        # centered coordinate system of the second camera:
        print(pose[0])
        print(x)
        print(pose[1])
        x_prim = pose[0] @ x[:3].T + pose[1]
        #3.- Return (R,t) si x3 > y x3'>0
        if x[-1] > 0 and x_prim[-1] > 0 : #Comprobamos que la profundidad es positiva
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
#F = cargar_matriz_F()#

#7.- Convertir en coordenadas homogeneas
pts_left,pts_right=conv_homogeneas(pts_left, pts_right)

i = np.random.randint(len(pts_left))

#8.- Aplicar el cóigo de Rectificación 
#Rectificacion = rectificacion_Esteroscipica_calibrada(E,pts_left[i],pts_right[i])

print("Rectificacion")
#print(Rectificacion)


#B = np.linalg.norm(Rectificacion[1])
#print(B)

'''
rec = np.hstack((Rectificacion[0],Rectificacion[1].reshape(3, 1)))
print(rec)
np.save('matriz_Rec.npy', rec)
'''
#9.- Correguimos signos negativos de la diagonal
K = cargar_matriz_K()
R = cargar_matriz_R()
t = cargar_matriz_t()
"""
D = np.diag(np.sign(np.diag(K)))
K = K @ D
print(img_left.shape)
print("K")
print(K)
print(np.linalg.det(K)) #Debe de dar un valor muy alto
"""

#Pasos pag 363 IREG.pdf

#Cargar F a memoria
F = cargar_matriz_F()
E = cargar_matriz_E()

#Determinal el epiolo eL desde F
U,S,Vt = np.linalg.svd(E)
e_L = Vt[-1]
e_L = e_L / e_L[2]

#Determinar matriz M 10.42
Sp = np.array([[0,-S[1],0],[-S[0],0,0],[0,0,1]])
M = U@Sp@Vt

#Determinar Ttrans de y0 != de los epipolos ec 20.33
width, height = img_left.shape[1], img_left.shape[0]
u0 = width / 2
v0 = height / 2
y0 = np.array([u0, v0])

trans = np.array([[1,0,-u0],[0,1,-v0],[0,0,1]])


#Determinar la translacion del epipolo e'L = ttrans @ eL
e0L = trans @ e_L

#Determinar la rotación de Trot desde  e'L ec 20.35
def rotation_matrix_to_x_axis(e):
        ex, ey = e[0], e[1]
        theta = np.arctan2(ey, ex)
        if abs(theta) > np.pi / 2:
            theta += np.pi  # Step 7: Optional flip
        cos_theta, sin_theta = np.cos(-theta), np.sin(-theta)
        return np.array([
            [cos_theta, sin_theta, 0],
            [-sin_theta,  cos_theta, 0],
            [0, 0, 1]
        ])

Trot = rotation_matrix_to_x_axis(e0L)


#êl desde aqui obtener Hinf ex 20.37
ehatL = Trot @ e0L
#Obtener la homografia HL
def compute_Hinf(ehatL):
        fu = ehatL[0]
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [-1 / fu, 0, 1]
        ])

Hinf = compute_Hinf(ehatL)
#Transformar todos los puntos de la imagen tanto iz como der
HL = Hinf @ Trot @ trans

    # Step 11: Transform points
def apply_homography(H, pts):
    pts = np.atleast_2d(pts)
    if pts.shape[1] == 3:
        pts = pts[:, :2]
    print(pts.shape)
    print(pts)
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    pts_t = (H @ pts_h.T).T
    return pts_t[:, :2] / pts_t[:, 2:]

yL_tilde = apply_homography(HL, pts_left)
yR_tilde = apply_homography(HL @ M, pts_right)

# Step 12: Form matrix YL̃ (canonical homogeneous coords)
YL_tilde = np.hstack([yL_tilde, np.ones((len(yL_tilde), 1))]).T

# Step 13: Form vector of u-coordinates from yR̃
uR_tilde = yR_tilde[:, 0]
print(yL_tilde)
uL_tilde = yL_tilde[:, 0]

# Step 14: Solve normal equations for a = (a, b, c)
YR_tilde = YL_tilde.T
a_vec = np.linalg.lstsq(YR_tilde, uR_tilde, rcond=None)[0]
#J= YR_tilde @ YR_tilde.T    # n x n matriz de rango 1
#print(YR_tilde)
#print(uL_tilde)
#Ñ = YR_tilde @ uL_tilde.T    # n x m

# Calcula pseudoinversa de M
#M_pinv = np.linalg.pinv(M)


# Step 15: Form A using a = (a, b, c)
#a_vec = M_pinv @ Ñ
a, b, c = a_vec
A = np.array([
    [a, b, c],
    [0, 1, 0],
    [0, 0, 1]
])

# Step 16: Compute HR
HR = A @ HL @ M



'''
# 10.- Normalizamos el vector de translacion
r1 = t / np.linalg.norm(t)

# 11.- Definimos un eje global
ez = np.array([0, 0, 1])

# 12.- Producto entre el eje y el vector de translacion y lo normalizamos
r2 = np.cross(ez, r1)
r2 = r2 / np.linalg.norm(r2)

# 13.- Eje Z rectificado
r3 = np.cross(r1, r2)

# 14.- Matriz de rotación
R_rect = np.vstack([r1, r2, r3])

# 15.- Rotaciones para cada camara, la derecha y la izquierda
R1 = R_rect
R2 = R @ R_rect 

#P1 = K @ np.array([np.eye(3,3),np.zeros(3,1)])
P1 = K @ np.hstack((np.eye(3,3), np.zeros((3,1))))
tpri = -R2 @ t

R22 = np.hstack((np.eye(3), tpri.reshape(3,1)))
P2 = K @ R22

print("a")
print(R1)

print(R2)

# 16.- Calculamos las homografias
K_inv = np.linalg.inv(K)
Hl = K @ R1.T @ K_inv
Hr = K @ R2.T @ K_inv

print(Hl)


print(Hr)
# 17.- Normalizar las homografias
Hl = Hl / Hl[2,2]
Hr = Hr / Hr[2,2]



# 10.- Normalizamos el vector de translacion
r1 = Rectificacion[1] / np.linalg.norm(Rectificacion[1])

# 11.- Definimos un eje global
ez = np.array([0, 0, 1])

# 12.- Producto entre el eje y el vector de translacion y lo normalizamos
r2 = np.cross(ez, r1)
r2 = r2 / np.linalg.norm(r2)

# 13.- Eje Z rectificado
r3 = np.cross(r1, r2)

# 14.- Matriz de rotación
R_rect = np.vstack([r1, r2, r3])

# 15.- Rotaciones para cada camara, la derecha y la izquierda
R1 = R_rect
R2 = Rectificacion[0] @ R_rect 

P1 = K @ np.array([np.eye(3,3),np.zero(3,1)])
tpri = -R2 @ Rectificacion[1]
P1 = K @ np.array([np.eye(3,3),tpri])

print("a")
print(R1)

print(R2)

# 16.- Calculamos las homografias
K_inv = np.linalg.inv(K)
Hl = K @ R1.T @ K_inv
Hr = K @ R2.T @ K_inv

print(Hl)


print(Hr)
# 17.- Normalizar las homografias
Hl = Hl / Hl[2,2]
Hr = Hr / Hr[2,2]
'''
# 18.- Las aplicamos sobre las imagenes
img_left_rect = cv2.warpPerspective(imgI, HL, (img_left.shape[1], img_left.shape[0]))
img_right_rect = cv2.warpPerspective(imgD, HR, (img_right.shape[1], img_right.shape[0]))

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