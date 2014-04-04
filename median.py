import scipy.optimize
import numpy as np
from sympy.utilities.lambdify import lambdify
import sympy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dim = 10

points = np.r_[np.random.random((11,dim)), np.random.random((10,dim)) + 3*np.ones((dim,))]

N, dim = points.shape

Xs = sympy.symbols(['X%d' % d for d in range(1, dim+1)])

distances = sum(sympy.sqrt(sum((xi - Xi)**2 for xi, Xi in zip(xs, Xs))) for xs in points)

jac = [sympy.diff(distances, Xi) for Xi in Xs]

#print(distances)

f_distances = lambdify(Xs, distances, modules='numpy')
f_jac = [lambdify(Xs, jac[i], modules='numpy') for i, Xi in enumerate(Xs)]

def jac(xs):
    return np.array([f_jac_i(*xs) for f_jac_i in f_jac])

res = scipy.optimize.minimize(lambda xs: f_distances(*xs),
        x0=points.sum(0)/points.shape[0], method='CG', jac=jac, callback=lambda *x: print(x))

#print("Result:")
#print(res)

r = np.linspace(0, 4, 20)
xs, ys = np.meshgrid(r, r)

fig = plt.figure()
if dim == 2:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1])
    ax.scatter(res['x'][0], res['x'][1], f_distances(*res['x']))
    ax.plot_wireframe(X=xs, Y=ys, Z=f_distances(xs, ys))
    #plt.plot(np.linspace(0,1,100), f_distances(res['x'][0], np.linspace(0, 1, 100)))
    #plt.plot(np.linspace(0,1,100), f_distances(np.linspace(0, 1, 100), res['x'][1]))
elif dim == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.scatter(res['x'][0], res['x'][1], res['x'][2], c='y')

plt.show()
