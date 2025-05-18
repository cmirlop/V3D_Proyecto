import numpy as np
import cv2
import glob

# Configuración del patrón del tablero
def config_tablero(dim, tamaño):
    pattern_size = dim  # Cambia a las dimensiones de tu tablero
    square_size = tamaño / 1000 # Tamaño real de cada cuadrado en metros (31mm)
    return  pattern_size,square_size 



#---- Pasos ---#

#1.- Configuramos las dimensiones del tablero, y el tamaño de cada cuadrado
pattern_size,square_size = config_tablero((7, 5),31.0)


# Preparar puntos 3D del tablero de ajedrez
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

# Listas para almacenar puntos 3D y puntos de imagen
object_points = []  # Puntos 3D en el espacio real
image_points = []   # Puntos 2D en el plano de la imagen

# Cargar las imágenes
image_paths = glob.glob('data/my_frame-*.jpg')
images = []
for path in image_paths:
    img = cv2.imread(path)
    
    if img is None:
        print(f"No se pudo cargar la imagen en {path}")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Intentar detectar las esquinas del tablero de ajedrez
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # Refinar las esquinas detectadas
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Dibujar las esquinas detectadas
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)

        # Mostrar la imagen con las esquinas
        cv2.imshow('Esquinas Detectadas', img)
        cv2.waitKey(500)  # Mostrar durante 500ms para cada imagen

        object_points.append(objp)
        image_points.append(corners)
        images.append(img)
        print(f"Se detectaron {len(corners)} esquinas en la imagen {path}")
    else:
        print(f"No se detectaron esquinas en la imagen {path}")

# Verificar si se detectaron suficientes puntos
if len(object_points) == 0 or len(image_points) == 0:
    raise ValueError("No se detectaron suficientes puntos de calibración.")

# Calcular la matriz de proyección
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

if not ret:
    print("Calibración fallida")
else:
    print(f"Calibración exitosa")

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