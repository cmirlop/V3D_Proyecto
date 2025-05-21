import numpy as np
import open3d as o3d

# --------------------------------------------------------------------------------------------------------

def reproject_image_to_3D(disparity, Q):
    height, width = disparity.shape
    points_3D = np.zeros((height, width, 3), dtype=np.float32)

    for y in range(height):
        for x in range(width):
            d = disparity[y, x] if disparity[y, x] > 0 else 0.0001  # Evita la division por 0
            vec = np.array([x, y, d, 1], dtype=np.float32)  # Pasa a coordenadas homogeneas
            point = Q @ vec  
            points_3D[y, x] = point[:3] / point[3]  # Normaliza con W

    return points_3D

# --------------------------------------------------------------------------------------------------------

def render(path):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])

# --------------------------------------------------------------------------------------------------------

def median_blur(image, ksize):
    if ksize % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser un número impar.")

    height, width = image.shape
    padded_image = np.pad(image, ksize // 2, mode='reflect')
    output = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            # Extrae la vecindad del píxel
            neighborhood = padded_image[i:i+ksize, j:j+ksize]
            # Calcula la mediana y la asigna al píxel correspondiente
            output[i, j] = np.median(neighborhood)

    return output

# --------------------------------------------------------------------------------------------------------
# Guarda la nube de puntos 3D con los colores en un archivo PLY
def save_point_cloud(filename, disparity, colors):
    Q = np.array([[1, 0, 0, -disparity.shape[1]/2],
                [0, -1, 0, disparity.shape[0]/2],
                [0, 0, 0, -0.8*disparity.shape[1]],  # distancia focal; adjust based on calibration
                [0, 0, 1/0.05, 0]])  # linea de base (adjust based on your setup)

    #points_3d = cv.reprojectImageTo3D(disparity, Q)
    points_3d = reproject_image_to_3D(disparity, Q)
    mask = disparity > 0  # Elimina los puntos en los que la disparidad en 0 o menos
    points = points_3d[mask]
    colors = colors[mask]
    points = np.hstack([points, colors])
    header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    with open(filename, "w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%f %f %f %d %d %d")
        
# --------------------------------------------------------------------------------------------------------
        