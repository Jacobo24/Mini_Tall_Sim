import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animar_temperatura(L=1.0, N=10, alpha=1.0, t_max=10):
    """
    Anima la evolución de la temperatura en 1D con condiciones:
      - Condición inicial: T(x,0) = 10*sin(pi*x)
      - Condiciones de frontera: T(0,t)=T(1,t)=0
    usando el método explícito de diferencias finitas.
    
    Parámetros:
      L: longitud del dominio (0 a L)
      N: número de nodos (incluyendo extremos)
      alpha: difusividad térmica
      t_max: tiempo total de simulación
    """
    # Cálculo del espaciamiento
    dx = L / (N - 1)
    # Para estabilidad: lambda = alpha*dt/dx^2 <= 0.5
    dt = 0.5 * dx**2 / alpha
    lam = alpha * dt / dx**2

    # Número total de pasos de tiempo
    n_steps = int(t_max / dt)

    # Dominio espacial
    x = np.linspace(0, L, N)
    
    # Condición inicial: T(x,0) = 10*sin(pi*x)
    T = 10 * np.sin(np.pi * x)
    # Las condiciones de frontera ya se satisfacen, pues sin(0)=0 y sin(pi)=0:
    T[0] = 0
    T[-1] = 0

    # Configuración de la figura para la animación
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 11)  # Ajustamos límites para ver la evolución
    ax.set_title("Evolución de la temperatura con T(x,0)=10*sin(pi*x)")
    ax.set_xlabel("Posición x (m)")
    ax.set_ylabel("Temperatura (°C)")
    
    # Línea que se actualizará en la animación
    (line,) = ax.plot(x, T, marker='o', color='blue')

    def update(frame):
        nonlocal T
        T_old = T.copy()
        # Actualizamos los nodos interiores con el esquema explícito
        for i in range(1, N-1):
            T[i] = T_old[i] + lam * (T_old[i+1] - 2*T_old[i] + T_old[i-1])
        # Se reimponen las condiciones de frontera
        T[0] = 0
        T[-1] = 0
        line.set_ydata(T)
        return line,

    # Crear la animación
    ani = FuncAnimation(fig, update, frames=range(n_steps), interval=200, blit=False)
    plt.show()
    return ani

# Llamada a la función de animación
ani = animar_temperatura()