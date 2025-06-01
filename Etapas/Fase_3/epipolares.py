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