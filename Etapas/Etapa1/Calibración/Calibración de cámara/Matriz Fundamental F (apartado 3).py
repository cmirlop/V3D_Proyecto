import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cargar las imágenes y convertirlas a escala de grises
image_left = cv2.imread('data/left.png') 
image_right = cv2.imread('data/right.png')

image_left_gray = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
image_right_gray = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)

# Detectar puntos clave y descriptores con SIFT
sift = cv2.SIFT_create()
kp_left, des_left = sift.detectAndCompute(image_left_gray, None)
kp_right, des_right = sift.detectAndCompute(image_right_gray, None)

# Emparejar puntos clave usando BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_left, des_right, k=2)

# Aplicar el filtro de razón de Lowe
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Obtener puntos clave buenos
pts_left = np.float32([kp_left[m.queryIdx].pt for m in good_matches])
pts_right = np.float32([kp_right[m.trainIdx].pt for m in good_matches])

# Calcular la matriz fundamental
F, mask = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)

# Filtrar inliers
pts_left = pts_left[mask.ravel() == 1]
pts_right = pts_right[mask.ravel() == 1]

# Crear una imagen combinada
combined_image = np.hstack((image_left_gray, image_right_gray))

# Visualización mejorada
plt.figure(figsize=(15, 5))
plt.imshow(combined_image, cmap='gray')

# Dibujar líneas epipolares
for pt1, pt2 in zip(pts_left, pts_right):
    pt2_shifted = (pt2[0] + image_left_gray.shape[1], pt2[1])
    plt.scatter(pt1[0], pt1[1], color='cyan', marker='x')
    plt.plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], color='yellow')

plt.show()

# Exportar y mostrar la matriz fundamental
print("Matriz Fundamental:")
print(F)

# Guardar la matriz fundamental en un archivo .npy
np.save('matriz_F.npy', F)