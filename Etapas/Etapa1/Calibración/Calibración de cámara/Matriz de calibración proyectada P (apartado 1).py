import numpy as np
import cv2
import glob

# Configuración del patrón del tablero
pattern_size = (9, 6)  # Cambia a las dimensiones de tu tablero
square_size = 1.0  # Tamaño real de cada cuadrado (ajusta según tu tablero)

# Preparar puntos 3D del tablero de ajedrez (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Listas para almacenar puntos 3D y puntos de imagen
object_points = []  # Puntos 3D en el espacio real
image_points = []   # Puntos 2D en el plano de la imagen

# Cargar las imágenes
image_paths = glob.glob('data/*.png')
images = []
for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        object_points.append(objp)
        image_points.append(corners)
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        images.append(img)
        print(f"Se detectaron {len(corners)} esquinas en la imagen {path}")
    else:
        print(f"No se detectaron esquinas en la imagen {path}")

# Verificar si se detectaron suficientes puntos
if len(object_points) == 0 or len(image_points) == 0:
    raise ValueError("No se detectaron suficientes puntos de calibración.")

# Calcular la matriz de proyección
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Construir la matriz de proyección P (3x4)
R, _ = cv2.Rodrigues(rvecs[0])
t = tvecs[0]
P = np.hstack((R, t))
P = np.dot(mtx, P)

# Normalizar la matriz de proyección
P /= P[-1, -1]

print("\nMatriz de proyección de la cámara (3x4):")
print(P)

# Calcular el error de reproyección
total_error = 0
for i in range(len(object_points)):
    imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print(f"\nError de reproyección: {total_error / len(object_points):.6f} píxeles")

# Guardar la matriz P en un archivo .npy
np.save('matriz_P.npy', P)
print("Matriz guardada correctamente.")

cv2.destroyAllWindows()
