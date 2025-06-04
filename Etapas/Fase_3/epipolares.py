import numpy as np
import matplotlib.pyplot as plt

def dibujar_epipolar(imagen_izquierda, imagen_derecha, F):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(imagen_izquierda)
    ax2.imshow(imagen_derecha)
    ax1.set_title('Imagen Izquierda')
    ax2.set_title('Imagen Derecha')

    def on_click(event):
        if event.inaxes != ax1:
            return
        
        x, y = event.xdata, event.ydata
        punto = np.array([x, y, 1])

        linea = F @ punto
        a, b, c = linea

        x_vals = np.array([0, imagen_derecha.shape[1]])
        y_vals = -(a * x_vals + c) / b
        ax2.plot(x_vals, y_vals, 'r')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def dibujar_epipolar_inv(imagen_izquierda, imagen_derecha, F):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(imagen_izquierda)
    ax2.imshow(imagen_derecha)
    ax1.set_title('Imagen Izquierda')
    ax2.set_title('Imagen Derecha')

    def on_click(event):
        if event.inaxes != ax2:
            return
        
        x, y = event.xdata, event.ydata
        punto = np.array([x, y, 1])

        linea = F.T @ punto
        a, b, c = linea

        x_vals = np.array([0, imagen_izquierda.shape[1]])
        y_vals = -(a * x_vals + c) / b
        ax1.plot(x_vals, y_vals, 'b')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def validar_epipolaridad(F, puntos1, puntos2):
    errores = []
    for p1, p2 in zip(puntos1, puntos2):
        x1 = np.array([p1[0], p1[1], 1.0])
        x2 = np.array([p2[0], p2[1], 1.0])
        error = abs(x2 @ F @ x1)
        errores.append(error)
    return np.array(errores)

def dibujar_epipolar_esencial(imagen_izq, imagen_der, E, K):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(imagen_izq)
    ax2.imshow(imagen_der)
    ax1.set_title('Imagen Izquierda')
    ax2.set_title('Imagen Derecha')

    K_inv = np.linalg.inv(K)

    def on_click(event):
        if event.inaxes != ax1:
            return

        x, y = event.xdata, event.ydata
        punto_img = np.array([x, y, 1.0])
        # Llevar a coordenadas normalizadas
        punto_norm = K_inv @ punto_img

        # Línea epipolar en la derecha en coordenadas normalizadas
        linea_norm = E @ punto_norm

        # Llevar la línea epipolar a coordenadas de imagen
        linea_img = np.linalg.inv(K).T @ linea_norm

        a, b, c = linea_img
        x_vals = np.array([0, imagen_der.shape[1]])
        y_vals = -(a * x_vals + c) / b
        ax2.plot(x_vals, y_vals, 'g')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def dibujar_epipolar_esencial_inv(imagen_izq, imagen_der, E, K):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(imagen_izq)
    ax2.imshow(imagen_der)
    ax1.set_title('Imagen Izquierda')
    ax2.set_title('Imagen Derecha')

    K_inv = np.linalg.inv(K)

    def on_click(event):
        if event.inaxes != ax2:
            return

        x, y = event.xdata, event.ydata
        punto_img = np.array([x, y, 1.0])
        punto_norm = K_inv @ punto_img

        # Línea epipolar en la izquierda en coordenadas normalizadas
        linea_norm = E.T @ punto_norm

        # Llevar la línea epipolar a coordenadas de imagen
        linea_img = np.linalg.inv(K).T @ linea_norm

        a, b, c = linea_img
        x_vals = np.array([0, imagen_izq.shape[1]])
        y_vals = -(a * x_vals + c) / b
        ax1.plot(x_vals, y_vals, 'g')
        fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()