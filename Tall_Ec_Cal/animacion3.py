import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animar_temperatura_dirichlet_neumann(L=1.0, N=10, alpha=1.0, T_left=100, T_init=20, t_max=1000):
    """
    Anima la evolución de la temperatura en 1D con:
      - Condición de Dirichlet en el extremo izquierdo (x=0): T = T_left (fijo).
      - Condición de Neumann en el extremo derecho (x=L): dT/dx = 0.
      
    Se actualizan los nodos interiores con el esquema explícito:
      T_i^(n+1) = T_i^n + λ (T_(i+1)^n - 2*T_i^n + T_(i-1)^n)
    y para el último nodo (i = N-1):
      T_(N-1)^(n+1) = T_(N-1)^n + 2λ (T_(N-2)^n - T_(N-1)^n)
    
    Parámetros:
      L: Longitud del dominio (0 a L).
      N: Número total de nodos.
      alpha: Difusividad térmica.
      T_left: Temperatura fija en x=0.
      T_init: Temperatura inicial en el resto del dominio.
      t_max: Tiempo total de simulación.
    """
    # Discretización espacial y temporal (condición de estabilidad: λ = alpha*dt/dx^2 <= 0.5)
    dx = L / (N - 1)
    dt = 0.5 * dx**2 / alpha
    lam = alpha * dt / dx**2

    # Número total de pasos de tiempo
    n_steps = int(t_max / dt)

    # Vector de posiciones
    x = np.linspace(0, L, N)

    # Estado inicial de la temperatura: T_init en el interior, T_left en el extremo izquierdo
    T = np.ones(N) * T_init
    T[0] = T_left

    # Configuración de la figura
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(T_init - 5, T_left + 5)
    ax.set_title("Evolución de la temperatura (Dirichlet en x=0, Neumann en x=L)")
    ax.set_xlabel("Posición x (m)")
    ax.set_ylabel("Temperatura (°C)")
    (line,) = ax.plot(x, T, marker='o', color='red')

    def update(frame):
        nonlocal T
        T_old = T.copy()
        # Actualización de nodos interiores (1 a N-2)
        for i in range(1, N-1):
            T[i] = T_old[i] + lam * (T_old[i+1] - 2*T_old[i] + T_old[i-1])
        # El extremo izquierdo se mantiene fijo:
        T[0] = T_left
        # Actualización del extremo derecho con condición de Neumann (flujo nulo)
        T[-1] = T_old[-1] + 2*lam*(T_old[-2] - T_old[-1])
        
        line.set_ydata(T)
        return (line,)

    # Creación de la animación; ajusta 'interval' según la velocidad deseada
    ani = FuncAnimation(fig, update, frames=range(n_steps), interval=100, blit=False)
    plt.show()
    return ani

# Llamada a la función de animación
ani = animar_temperatura_dirichlet_neumann()