import numpy as np
import sympy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from scipy import constants


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __radd__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __rsub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar):
        return Vector(self.x / scalar, self.y / scalar)


class Body:
    i = 1

    def __init__(self, m, pos0, vel0):
        Body.i += 1

        self.m = m
        self.G = 1 # constants.gravitational_constant

        self.pos = Vector(sympy.symbols(f"x{self.i}"), sympy.symbols(f"y{self.i}"))
        self.vel = Vector(sympy.symbols(f"vx{self.i}"), sympy.symbols(f"vy{self.i}"))
        self.a = Vector(0, 0)
        self.pos0 = pos0
        self.vel0 = vel0

        self.vel_function = Vector([], [])
        self.a_function = Vector([], [])
        self.pos_integrated = Vector([], [])
        self.vel_integrated = Vector([], [])

    def calculate_a(self, bodies):
        if self == bodies[0]:
            other1 = bodies[1]
            other2 = bodies[2]
        elif self == bodies[1]:
            other1 = bodies[0]
            other2 = bodies[2]
        else:
            other1 = bodies[0]
            other2 = bodies[1]
        self.a = -self.G * (other1.m * (self.pos - other1.pos) /
                            (((self.pos.x - other1.pos.x)**2 + (self.pos.y - other1.pos.y)**2)**(3/2))
                            + other2.m * (self.pos - other2.pos) /
                            (((self.pos.x - other2.pos.x)**2 + (self.pos.y - other2.pos.y)**2)**(3/2)))

    def create_vel_function(self):
        self.vel_function.x = sympy.lambdify(self.vel.x, self.vel.x)
        self.vel_function.y = sympy.lambdify(self.vel.y, self.vel.y)

    def create_a_function(self, bodies):
        conditions = [bodies[0].pos.x, bodies[0].pos.y, bodies[1].pos.x, bodies[1].pos.y, bodies[2].pos.x, bodies[2].pos.y]
        self.a_function.x = sympy.lambdify([conditions], self.a.x)
        self.a_function.y = sympy.lambdify([conditions], self.a.y)


bodies = [
    Body(2, Vector(1, 2), Vector(-0.25, -0.5)),
    Body(2, Vector(-0.75, 1), Vector(0.45, 1)),
    Body(2, Vector(-0.5, -1), Vector(0, 0.425))
]
for body in bodies:
    body.calculate_a(bodies)
    body.create_vel_function()
    body.create_a_function(bodies)


# function that returns dz/dt
def model(y0, t):
    pos_equations = y0[0:6]
    create_vel_functions = y0[6:12]
    dzdt = []
    for body in bodies:
        dzdt.append(body.vel_function.x(create_vel_functions[2 * bodies.index(body)]))
        dzdt.append(body.vel_function.y(create_vel_functions[2 * bodies.index(body) + 1]))
    for body in bodies:
        dzdt.append(body.a_function.x(pos_equations))
        dzdt.append(body.a_function.y(pos_equations))
    return dzdt


# initial conditions
y0 = [bodies[0].pos0.x, bodies[0].pos0.y, bodies[1].pos0.x, bodies[1].pos0.y, bodies[2].pos0.x, bodies[2].pos0.y,
      bodies[0].vel0.x, bodies[0].vel0.y, bodies[1].vel0.x, bodies[1].vel0.y, bodies[2].vel0.x, bodies[2].vel0.y]

# time points
t = np.linspace(0, 60, 801)

# solve ODE
solution = np.transpose(odeint(model, y0, t))

for body in bodies:
    body.pos_integrated.x = solution[2*bodies.index(body)]
    body.pos_integrated.y = solution[2*bodies.index(body)+1]

fig, ax = plt.subplots()


def animate(i):
    ax.clear()
    ax.set_xlim(-2.5, 20)
    ax.set_ylim(-2.5, 20)
    body1_path, = ax.plot(bodies[0].pos_integrated.x[0:i], bodies[0].pos_integrated.y[0:i], color='orange', lw=1)
    body2_path, = ax.plot(bodies[1].pos_integrated.x[0:i], bodies[1].pos_integrated.y[0:i], color='green', lw=1)
    body3_path, = ax.plot(bodies[2].pos_integrated.x[0:i], bodies[2].pos_integrated.y[0:i], color='blue', lw=1)
    body1_point, = ax.plot(bodies[0].pos_integrated.x[i], bodies[0].pos_integrated.y[i], marker='.', color='orange')
    body2_point, = ax.plot(bodies[1].pos_integrated.x[i], bodies[1].pos_integrated.y[i], marker='.', color='green')
    body3_point, = ax.plot(bodies[2].pos_integrated.x[i], bodies[2].pos_integrated.y[i], marker='.', color='blue')
    return body1_path, body2_path, body3_path, body1_point, body2_point, body3_point,


if not os.path.exists("gifs"):
    os.mkdir("gifs")
ani = FuncAnimation(fig, animate, interval=240, blit=True, repeat=True, frames=799)
ani.save("gifs/threebodyproblem.gif", dpi=300, writer=PillowWriter(fps=30))
