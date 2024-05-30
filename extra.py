import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def booth_function(arreglo):
    x = arreglo[0] 
    y = arreglo[1]
    return ((x + 2*y - 7)**2) + ((2*x + y - 5)**2)

def actualizar_delta(delta,alpha):
    lista_deltas = []
    for i in delta:
        nueva_delta = i / 2
        lista_deltas.append(nueva_delta)
    return lista_deltas

def distancia_origen(vector):
    return np.linalg.norm(vector)

def verificar_distancia(vector, e):
    distancia = distancia_origen(vector)
    return distancia < e








def exploratory_move(x, deltas, objective_function):
    n = len(x)
    best_x = x[:]
    best_value = objective_function(x)
    for i in range(n):
        x_new = x[:]
        x_new[i] += deltas[i]
        new_value = objective_function(x_new)
        if new_value < best_value:
            best_x = x_new[:]
            best_value = new_value
        x_new = x[:]
        x_new[i] -= deltas[i]
        new_value = objective_function(x_new)
        if new_value < best_value:
            best_x = x_new[:]
            best_value = new_value
    return best_x

def pattern_move(xk, xk_1):
    return [xk[i] + (xk[i] - xk_1[i]) for i in range(len(xk))]


def hooke_jeeves(x0, deltas, alpha, epsilon, objective_function):
    xk = x0[:]
    xk_1 = x0[:]
    k = 0
    exploratorios = [] 
    patron = []

    distancia = distancia_origen(deltas)
    
    while (distancia > 0):
        xk_new = exploratory_move(xk, deltas, objective_function)
        exploratorios.append(xk_new[:])
        if xk_new != xk:
            xk = xk_new[:]
            xk_1 = xk[:]
            k += 1
            
            xk_p = pattern_move(xk, xk_1)
            exploratorios.append(xk_p[:])
            
            xk_new = exploratory_move(xk_p, deltas, objective_function)
            exploratorios.append(xk_new[:])
            
            if objective_function(xk_new) < objective_function(xk):
                xk = xk_new[:]
            else:
                if max(deltas) < epsilon:
                    break
                deltas = [delta / alpha for delta in deltas]
        else:
            if max(deltas) < epsilon:
                break
            deltas = [delta / alpha for delta in deltas]
    
    return xk, exploratorios



def objective_function(x):
    # Funcion Sphere
    return np.sum(np.square(x))

    # Funcion Himmelblau
    # return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2




# Funcion Sphere
# x0 = [-1.0, 1.5]

# Funcion Himmelblau
# x0 = [3.0, 2.0]


x0 = [5, 5] 
deltas = [0.5,0.5]  
alpha = 2  
epsilon = 1e-6 



optimo, exploratorios = hooke_jeeves(x0, deltas, alpha, epsilon, booth_function)
print("Resultado final:", optimo)
print("Valor de la función objetivo en el resultado final:", booth_function(optimo))


# optimo, exploratorios = hooke_jeeves(x0, deltas, alpha, epsilon, objective_function)
# print("Resultado final:", optimo)
# print("Valor de la función objetivo en el resultado final:", objective_function(optimo))


fig, ax = plt.subplots()
x_data, y_data = [], []

x = np.linspace(-8, 8, 400)
y = np.linspace(-8, 8, 400)
X, Y = np.meshgrid(x, y)
Z = booth_function([X, Y])

contour = ax.contourf(X, Y, Z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Booth Function')

point, = ax.plot([], [], 'bo')
path, = ax.plot([], [], 'r-', alpha=0.5)

def init():
    point.set_data([], [])
    path.set_data([], [])
    return point, path

def update(frame):
    x_data.append(exploratorios[frame][0])
    y_data.append(exploratorios[frame][1])
    # point.set_data(exploratorios[frame][0], exploratorios[frame][1])
    point.set_data([exploratorios[frame][0]], [exploratorios[frame][1]])

    path.set_data(x_data, y_data)
    return point, path

ani = animation.FuncAnimation(fig, update, frames=len(exploratorios), init_func=init, blit=True, repeat=False)

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Hooke-Jeeves Exploratory Moves')
plt.grid()
plt.show()