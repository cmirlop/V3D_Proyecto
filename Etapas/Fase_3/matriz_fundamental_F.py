import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import convolve, gaussian_filter, maximum_filter
from scipy.signal import convolve2d

def harris(imagen, sigma):
    sobel = np.array([[-1, 0, 1]])
    #gaussian = int(2 * np.ceil(2 * sigma) + 1)

    Iu = convolve(imagen, sobel, mode='reflect')
    Iv = convolve(imagen, sobel.T, mode='reflect')

    Iuu = gaussian_filter(Iu ** 2, sigma)
    Ivv = gaussian_filter(Iv ** 2, sigma)
    Iuv = gaussian_filter(Iu * Iv, sigma)

    tr = Iuu + Ivv
    dt = Iuu * Ivv - Iuv ** 2

    C = dt - 0.04 * (tr ** 2)

    return C

def puntos_harris(C, umbral):
    #umbral = umbral * C.max()

    puntos_locales = (C == maximum_filter(C, size=7))
    detectados = (C > umbral) & puntos_locales
    y, x = np.nonzero(detectados)

    return np.array([x, y]).T

def extraer_parches(imagen, puntos, tamano_parche=11):
    mitad = tamano_parche // 2
    relleno = np.pad(imagen, mitad, mode='reflect')
    parches = []

    for x, y in puntos:
        x += mitad
        y += mitad
        parche = relleno[y - mitad:y + mitad + 1, x - mitad:x + mitad + 1]
        parche = parche - np.mean(parche)
        parche = parche / np.linalg.norm(parche)
        parches.append(parche)

    return np.array(parches)

def comparar_parches(parches1, parches2, ratio=0.75):
    coincidencias = []
    parches1 = parches1.reshape(parches1.shape[0], -1)
    parches2 = parches2.reshape(parches2.shape[0], -1)
    for i, parche1 in enumerate(parches1):
        parche1 = parche1.flatten()
        distancias = np.linalg.norm(parches2 - parche1, axis=1)
        if len(distancias) < 2:
            continue
        indices = np.argsort(distancias)
        d1 = distancias[indices[0]]
        d2 = distancias[indices[1]]

        # Criterio de Lowe
        if d1 < ratio * d2:
            coincidencias.append((i, indices[0]))

    return coincidencias

def dibujar_coincidencias(imagen1, imagen2, puntos1, puntos2):
    alto, ancho = imagen1.shape
    imagen_combinada = np.zeros((alto, ancho * 2), dtype=np.uint8)
    imagen_combinada[:, :ancho] = imagen1
    imagen_combinada[:, ancho:] = imagen2

    for (x1, y1), (x2, y2) in zip(puntos1, puntos2):
        cv2.circle(imagen_combinada, (x1, y1), 5, (255, 0, 0), -1)
        cv2.circle(imagen_combinada, (x2 + ancho, y2), 5, (255, 0, 0), -1)
        cv2.line(imagen_combinada, (x1, y1), (x2 + ancho, y2), (255, 0, 0), 1)

    plt.imshow(imagen_combinada, cmap='gray')
    plt.title('Coincidencias entre imágenes')
    plt.show()

def normalizar_puntos(puntos):
    media = np.mean(puntos, axis=0)
    std = np.std(puntos, axis=0)
    escala = np.sqrt(2) / std

    T = np.array([
        [escala[0], 0, -escala[0] * media[0]],
        [0, escala[1], -escala[1] * media[1]],
        [0, 0, 1]
    ])

    puntos_homogeneos = np.hstack((puntos, np.ones((puntos.shape[0], 1))))
    puntos_normalizados = (T @ puntos_homogeneos.T).T

    return puntos_normalizados, T

def algoritmo_8_puntos(puntos1, puntos2):
    puntos1_norm, T1 = normalizar_puntos(puntos1)
    puntos2_norm, T2 = normalizar_puntos(puntos2)

    A = []
    for i in range(puntos1.shape[0]):
        x1, y1 = puntos1_norm[i, :2]
        x2, y2 = puntos2_norm[i, :2]
        A.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])

    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F =  U @ np.diagflat(S) @ Vt

    F = T2.T @ F @ T1
    return F / F[2, 2]

def error_geometrico(puntos1, puntos2, F):
    puntos1_homogeneos = np.hstack((puntos1, np.ones((puntos1.shape[0], 1))))
    puntos2_homogeneos = np.hstack((puntos2, np.ones((puntos2.shape[0], 1))))

    l2 = (F @ puntos1_homogeneos.T).T
    l1 = (F.T @ puntos2_homogeneos.T).T

    d2 = np.abs(np.sum(l2 * puntos2_homogeneos, axis=1)) / np.linalg.norm(l2[:, :2], axis=1)
    d1 = np.abs(np.sum(l1 * puntos1_homogeneos, axis=1)) / np.linalg.norm(l1[:, :2], axis=1)

    return d1 + d2

def ransac(puntos1, puntos2, iteraciones=1000, umbral=1.0):
    mejor_F = None
    mejor_inliers = []

    for _ in range(iteraciones):
        indices = np.random.choice(puntos1.shape[0], 8, replace=False)
        F = algoritmo_8_puntos(puntos1[indices], puntos2[indices])

        errores = error_geometrico(puntos1, puntos2, F)
        inliers = np.where(errores < umbral)[0]

        if len(inliers) > len(mejor_inliers):
            mejor_inliers = inliers
            mejor_F = F

    if len(mejor_inliers) >= 8:
        mejor_F = algoritmo_8_puntos(puntos1[mejor_inliers], puntos2[mejor_inliers])

    return mejor_F, mejor_inliers

def visualizar_inliers(img1, img2, pts1, pts2, inliers):
    """
    Muestra las dos imágenes una al lado de la otra, conectando los puntos inliers.
    img1, img2: imágenes como arrays numpy (grises o RGB)
    pts1, pts2: coordenadas (N, 2)
    inliers: índices de los puntos válidos
    """
    if img1.ndim == 2:
        img1 = np.stack([img1]*3, axis=-1)
    if img2.ndim == 2:
        img2 = np.stack([img2]*3, axis=-1)

    # Crear imagen compuesta
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    altura = max(h1, h2)
    combinada = np.zeros((altura, w1 + w2, 3), dtype=np.uint8)
    combinada[:h1, :w1] = img1
    combinada[:h2, w1:] = img2

    # Mostrar
    plt.figure(figsize=(12, 8))
    plt.imshow(combinada)
    plt.axis('off')

    for i in inliers:
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        plt.plot([x1, x2 + w1], [y1, y2], 'r', linewidth=0.8)
        plt.scatter([x1, x2 + w1], [y1, y2], s=10, c='yellow')

    plt.title(f'Inliers encontrados: {len(inliers)}')
    plt.show()


#mostrar_puntos(imagen1, puntos1)
def mostrar_puntos(imagen, puntos):
    plt.imshow(imagen, cmap='gray')
    plt.scatter(puntos[:, 0], puntos[:, 1], c='red', s=5)
    plt.title('Puntos Harris detectados')
    plt.axis('off')
    plt.show()

#dibujar_coincidencias(imagen1, imagen2, puntos1, puntos2)