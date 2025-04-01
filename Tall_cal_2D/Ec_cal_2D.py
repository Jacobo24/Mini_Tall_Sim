import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parámetros del problema
alpha = 1.0         # Difusividad térmica
Lx, Ly = 1.0, 1.0   # Dimensiones del dominio
Nx, Ny = 50, 50     # Número de nodos en x y y
dx = Lx / Nx
dy = Ly / Ny

dt = 0.0001         # Paso de tiempo (verifica la condición de estabilidad)
Nt = 300            # Número de frames de la animación

# Condiciones de frontera
T_left   = 50.0     # Borde izquierdo (x = 0)
T_right  = 50.0     # Borde derecho (x = Lx)
T_bottom = 50.0     # Borde inferior (y = 0)
T_top    = 200.0    # Borde superior (y = Ly)

# Función de condición inicial: estado uniforme a 50 en todo el dominio
def T0(x, y):
    return 50.0

# Creamos la malla
x_vals = np.linspace(0, Lx, Nx+1)
y_vals = np.linspace(0, Ly, Ny+1)

# Inicializamos la matriz de temperaturas
T = np.zeros((Nx+1, Ny+1))
for i in range(Nx+1):
    for j in range(Ny+1):
        T[i, j] = T0(x_vals[i], y_vals[j])

# Aplicamos las condiciones de frontera
T[0, :]   = T_left     # Borde izquierdo
T[Nx, :]  = T_right    # Borde derecho
T[:, 0]   = T_bottom   # Borde inferior
T[:, Ny]  = T_top      # Borde superior

# Configuramos la figura y el eje para el mapa de calor
fig, ax = plt.subplots(figsize=(6,5))
# Usamos imshow con interpolación bilineal para suavizar la transición entre colores
im = ax.imshow(T, origin='lower', extent=[0, Lx, 0, Ly],
               cmap='jet', vmin=50, vmax=200, interpolation='bilinear')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Temperatura')
ax.set_title("Distribución de temperatura")
ax.set_xlabel('x')
ax.set_ylabel('y')

# Función de actualización para la animación
def update(frame):
    global T
    T_old = T.copy()
    
    # Actualizamos solo los nodos interiores
    for i in range(1, Nx):
        for j in range(1, Ny):
            d2T_dx2 = (T_old[i+1, j] - 2*T_old[i, j] + T_old[i-1, j]) / dx**2
            d2T_dy2 = (T_old[i, j+1] - 2*T_old[i, j] + T_old[i, j-1]) / dy**2
            T[i, j] = T_old[i, j] + alpha * dt * (d2T_dx2 + d2T_dy2)
    
    # Reaplicamos las condiciones de frontera en cada paso
    T[0, :]   = T_left
    T[Nx, :]  = T_right
    T[:, 0]   = T_bottom
    T[:, Ny]  = T_top

    im.set_data(T)
    ax.set_title(f"Distribución de temperatura\nt = {frame*dt:.4f}")
    return [im]

# Creamos la animación
anim = FuncAnimation(fig, update, frames=Nt, interval=50, blit=True)

plt.show()
