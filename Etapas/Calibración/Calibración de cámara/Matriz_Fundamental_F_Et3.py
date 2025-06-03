import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

#Función que lee las imagenes y las convierte a escala de grises
'''
def cargar_imagenes():
    # Cargar las imágenes y convertirlas a escala de grises
    image_left = cv2.imread('data/left2.png') 
    image_right = cv2.imread('data/right2.png')

    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    return image_left_gray, image_right_gray,image_left,image_right
'''
def cargar_imagenes(image_left,image_right):
    # Cargar las imágenes y convertirlas a escala de grises
    #image_left = cv2.imread('data/left2.png') 
    #image_right = cv2.imread('data/right2.png')

    image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    return image_left_gray, image_right_gray


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

    f2 = None
    inliners = []
    mascara = []
    #Aplicamos RANSAC
    for iter in range(10):
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
        U2,S2,Vt2 = np.linalg.svd(f2)
        S2[2] = 0  # Forzar el rango a 2
        f2 = U2 @ np.diag(S2) @ Vt2
        inliners2 = calcular_inliners(puntos_izq,puntos_drch,f2)
        mascara2 = calcular_mascara(puntos_izq,puntos_drch,f2)
        if len(inliners2) > len(inliners):
            inliners = inliners2
            mascara = mascara2
    
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

#Corrige la linea epipolar para que no se salga de la imagen
def correguir_linea_epipolar(a, b, c, width, height):
    points = []
    # Interseccion con el eje X
    for x in [0, width-1]:
        if b != 0:
            y = -(a*x + c)/b
            if 0 <= y < height:
                points.append((x, int(y)))

    # Intersección con el eje Y
    for y in [0, height-1]:
        if a != 0:
            x = -(b*y + c)/a
            if 0 <= x < width:
                points.append((int(x), y))
    # Selecciona solo dos puntos distintos
    if len(points) > 2:
        # Elimina duplicados
        points = list(dict.fromkeys(points))
    if len(points) >= 2:
        return points[:2]
    else:
        return None  # No hay segmento visible


def main_F(img_left, img_right):
    #--- Pasos ---#

    #1.- Leemos imagenes y las convertimos de color a gris
    img_left , img_right = cargar_imagenes(img_left, img_right)

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
    f2, inliners,mask = calcular_fundamental(pts_left, pts_right)
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
    np.save('inliners.npy', inliners)


    #Pasamos a dibujar las lineas epipolares
    #10.- Convertimos las coordenadas a homogenas
    pts_left_h = np.hstack([pts_left, np.ones((len(inliners), 1))])
    pts_right_h = np.hstack([pts_right, np.ones((len(inliners), 1))])

    np.save('pts1.npy', pts_left_h)
    np.save('pts2.npy', pts_right_h)

    #11.- Calculamos los puntos epipolares
    lines2 = (f2 @ pts_left_h.T).T
    lines1 = (f2.T @ pts_right_h.T).T


    #12.- Preapramos la figura para dibujar el epipolo
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    #Imagen izquierda
    
   
    axs[0].imshow(img_left)
    axs[0].set_title("Left image")
    axs[0].axis("off")
    for (a, b, c), (x, y,_) in zip(lines1, pts_left_h):
        pts = correguir_linea_epipolar(a, b, c, img_left.shape[1], img_left.shape[0])
        if pts is not None:
            (x0, y0), (x1, y1) = pts
            axs[0].plot([x0, x1], [y0, y1], 'k-', linewidth=1)
        axs[0].plot(x, y, 'wo', markersize=6, markeredgecolor='black')
    """for (a, b, c), (x, y,_) in zip(lines1, pts_left_h):
        x0, x1 = 0, img_left.shape[1]
        y0 = int(-(a * x0 + c) / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else img_left.shape[0]
        axs[0].plot([x0, x1], [y0, y1], 'k-', linewidth=1)
        axs[0].plot(x, y, 'wo', markersize=6, markeredgecolor='black')"""

    #Imagen derecha
    axs[1].imshow(img_right)
    axs[1].set_title("Right image")
    axs[1].axis("off")
    for (a, b, c), (x, y,_) in zip(lines2, pts_right_h):
        pts = correguir_linea_epipolar(a, b, c, img_right.shape[1], img_right.shape[0])
        if pts is not None:
            (x0, y0), (x1, y1) = pts
            axs[1].plot([x0, x1], [y0, y1], 'k-', linewidth=1)
        axs[1].plot(x, y, 'wo', markersize=6, markeredgecolor='black')
    """for (a, b, c), (x, y,_) in zip(lines2, pts_right_h):
        x0, x1 = 0, img_right.shape[1]
        y0 = int(-(a * x0 + c) / b) if b != 0 else 0
        y1 = int(-(a * x1 + c) / b) if b != 0 else img_right.shape[0]
        axs[1].plot([x0, x1], [y0, y1], 'k-', linewidth=1)
        axs[1].plot(x, y, 'wo', markersize=6, markeredgecolor='black')"""

    
    #13.- Calculamos los epipolos
    _, _, Vt = np.linalg.svd(f2)
    epipolo_d = Vt[-1] / Vt[-1][-1]  # En imagen derecha

    _, _, Vt_T = np.linalg.svd(f2.T)
    epipolo_i = Vt_T[-1] / Vt_T[-1][-1]  # En imagen izquierda

    #14.- Dibujamos el punto en las imágenes
    axs[0].plot(epipolo_i[0], epipolo_i[1], 'rx', markersize=10, label='Epipolo')
    axs[1].plot(epipolo_d[0], epipolo_d[1], 'rx', markersize=10, label='Epipolo')
    plt.tight_layout()
    plt.show()

    return f2, pts_left, pts_right



'''
def dibujar_lineas_epipolares(img1, img2, F, pts1, pts2):

        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        n = pts1.shape[0]
        for i in range(n):
            color = tuple(np.random.randint(0,255,3).tolist())
            # Linea epipolar para la imagen derecha
            x = np.array([pts2[i,0], pts2[i,1], 1])
            l = F @ x
            a, b, c = l
            # Draw the line on img2
            x0, y0 = 0, int(-c/b) if b != 0 else 0
            x1, y1 = img2.shape[1], int((-c - a*img2.shape[1])/b) if b != 0 else 0
            cv2.line(img2_color, (x0,y0), (x1,y1), color, 3)
            cv2.circle(img1_color, (int(pts2[i,0]), int(pts2[i,1])), 5, color, -1)

            # Lineas epipolares en la imagen izquierda
            x_ = np.array([pts1[i,0], pts1[i,1], 1])

            l_ = F.T @ x_ 
            a_, b_, c_ = l_
            x0_, y0_ = 0, int(-c_/b_) if b_ != 0 else 0
            x1_, y1_ = img1.shape[1], int((-c_ - a_*img1.shape[1])/b_) if b_ != 0 else 0
            cv2.line(img1_color, (x0_,y0_), (x1_,y1_), color, 3)
            cv2.circle(img2_color, (int(pts1[i,0]), int(pts1[i,1])), 5, color, -1)
        return img1_color, img2_color




    #15.- Dibujamos las lineas epipolares

img_left_epi, img_right_epi = dibujar_lineas_epipolares(img_left, img_right, f2, pts_left, pts_right)
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.title("Epipolar lines in left image")
plt.imshow(img_left_epi)
plt.subplot(1,2,2)
plt.title("Epipolar lines in right image")
plt.imshow(img_right_epi)
plt.show()
'''