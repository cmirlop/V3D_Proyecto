import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

#Función que lee las imagenes y las convierte a escala de grises
def cargar_imagenes():
    # Cargar las imágenes y convertirlas a escala de grises
    image_left = cv2.imread('data/left2.png') 
    image_right = cv2.imread('data/right2.png')

    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    return image_left_gray, image_right_gray,image_left,image_right


def calcular_inliners(pts_left, pts_right,F):
    inliners = []
    for pt_i, pt_d in zip(pts_left,pts_right):
        err = abs(pt_d.T @ F @ pt_i)
        if err < 0.01:
            inliners.append((pt_i,pt_d))
    return inliners

def calcular_mascara(pts_left, pts_right,F):
    mascara = []
    for pt_i, pt_d in zip(pts_left,pts_right):
        #Convertimso en homogeneas los puntos
        x_i = np.array([pt_i[0], pt_i[1], 1.0])  # izquierda
        x_d = np.array([pt_d[0], pt_d[1], 1.0])  # derecha
        #Calculamos su error absoluta
        err = abs(x_d @ F @ x_i)  # ya son vectores 1x3
        #Si este es menor a 0.01 lo guardamos en la mascara
        mascara.append(1 if err < 0.01 else 0)
    mascara = np.array(mascara).reshape(-1, 1)

    return mascara

# Detectar puntos clave y descriptores con SIFT
def apl_sift(img_left,img_right):
    sift = cv2.SIFT_create()
    kp_left, des_left = sift.detectAndCompute(img_left, None)
    kp_right, des_right = sift.detectAndCompute(img_right, None)
    return kp_left,des_left,kp_right,des_right

# Emparejar puntos clave usando BFMatcher
def apl_matcher(des_left, des_right):
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

# Filtrar inliers
def filtr_inliners(pts_left,pts_right,mask):
    pts_left = pts_left[mask.ravel() == 1]
    pts_right = pts_right[mask.ravel() == 1]
    return pts_left, pts_right

#Calcular la matriz fundamental
def calcular_fundamental(pts_left, pts_right) : 
    #Añadimos 1 a la 3a columnda para trabajar con coordenadas homogeneas
    puntos_izq = pts_left.copy()
    puntos_izq = np.hstack([puntos_izq, np.ones((puntos_izq.shape[0], 1))])

    puntos_drch = pts_right.copy()
    puntos_drch = np.hstack([puntos_drch, np.ones((puntos_drch.shape[0], 1))])

    # Contruimos A
    # A = xix2i, xiy2i xi yix2i yiy2i yi x2i y2i 1
    A = np.empty((0, 9))
    indices = np.random.choice(len(puntos_izq), size=8, replace=False)
    muestras_izq = puntos_izq[indices]
    #indices2 = np.random.choice(len(puntos_drch), size=8, replace=False)
    muestras_drch = puntos_drch[indices]

    for pt_izq, pt_dr, val in zip(muestras_izq, muestras_drch, range(0,8)):
        x1,y1,_ = pt_dr
        x2,y2,_2 = pt_izq
        fila = np.array([[x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]])
        A = np.vstack([A, fila])  # Apilar filas verticalmente
        if val == 8:  # El índice va de 0 a 7 para 8 elementos
            break
    
    U1, S1, Vt1 = np.linalg.svd(A)
    f2 = Vt1[-1, :]
    print(A@f2)
    f2 = f2.reshape(3, 3)
    f2 = f2 / f2[2,2]
    inliners = calcular_inliners(puntos_izq,puntos_drch,f2)
    mascara = calcular_mascara(puntos_izq,puntos_drch,f2)
    
    return f2,inliners,mascara


def mostrar_img_puntos_coincidentes(img_left, img_right, pts_left, pts_right):
    
    # Crear una imagen combinada
    combined_image = np.hstack((img_left, img_right))

    # Visualización mejorada
    plt.figure(figsize=(15, 5))
    plt.imshow(combined_image, cmap='gray')

    # Dibujar líneas de los puntos coincidentes
    for pt1, pt2 in zip(pts_left, pts_right):
        pt2_shifted = (pt2[0] + img_left.shape[1], pt2[1])
        plt.scatter(pt1[0], pt1[1], color='cyan', marker='x')
        plt.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color='yellow')

    plt.show()


#--- Pasos ---#

#1.- Leemos imagenes y las convertimos de color a gris
img_left , img_right, imgI, imgD = cargar_imagenes()

#2.- Aplicamos SIFT
kp_left, des_left,kp_right, des_right = apl_sift(img_left , img_right)

#3.- Aplicamos BFMatcher para emarejar puntos clave
matches = apl_matcher(des_left, des_right)

#4.- Aplicamos el filtro de lowe
good_matches = filtro_lowe(matches)


#5.- Obtener puntos clave buenos
pts_left, pts_right = obtener_puntos_buenos(kp_left,kp_right,good_matches)

#6.- Calcular la matriz fundamental
#F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)
f2,_,mask = calcular_fundamental(pts_left, pts_right)
print("Matriz Fundamental")
print(f2)
#print(mask)       
#print(pts_left.shape[0])

#7.- Filtramos inliners de los puntos que tenemos
pts_left,pts_right = filtr_inliners(pts_left,pts_right,mask)

#8.- Mostrar la imagen con los puntos coincidentes
mostrar_img_puntos_coincidentes(img_left, img_right, pts_left, pts_right)

#9.- Guardar la matriz fundamental en un archivo .npy
np.save('matriz_F.npy', f2)


'''

#Dibujar la linea epipolar


# Punto en la imagen izquierda
x = np.array(random.choice(pts_left))     # [45, 70]
x_homog = np.append(x, 1)  # en coordenadas homogéneas


# Calcula la línea epipolar l' = F * x
l = f2 @ x_homog  # Resultado: [a, b, c] tal que a*x + b*y + c = 0

# Para dibujar la línea: calcula dos puntos extremos (x=0 y x=ancho_imagen)
a, b, c = l
h, w = img_right.shape
pt1 = (0, int(-c / b)) if b != 0 else (0, 0)
pt2 = (w, int(-(a * w + c) / b)) if b != 0 else (w, 0)

# Dibuja la línea en la imagen derecha
img_right_line = img_right.copy()
cv2.line(img_right_line, pt1, pt2, (255, 0, 0), 2)

# Muestra la imagen
plt.imshow(img_right_line)
plt.title('Línea epipolar en imagen derecha')
plt.show()
'''