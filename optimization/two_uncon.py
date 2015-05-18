__author__ = 'chensn'

import numpy as np
from numpy import dot, array
import csv
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import theano
import theano.tensor as T


def loadData(file):
    dataset = []
    with open(file) as fin:
        reader = csv.reader(fin)
        for line in reader:
            dataset.append(line)
    dataset = np.array([[float(d) for d in row] for row in dataset[1:]])
    t = dataset[:, 0]
    a = dataset[:, 1]
    b = dataset[:, 2]
    c = dataset[:, 3]
    return t, a, b, c


# Backtracking-Armijo line search
def backtracking_armijo(x0, p, func, grad_func, dataset):
    alpha = 1
    rho = 0.25
    gamma = 0.1
    while not (func(x0 + alpha * p, dataset)) <= func(x0, dataset) + rho * np.dot(grad_func(x0, dataset), p) * alpha:
        # print alpha
        alpha *= gamma
    return alpha


def auto4check(dataset, x):

    t0 = theano.shared(value=dataset[0], name="t0")
    a0 = theano.shared(value=dataset[1], name="a0")
    b0 = theano.shared(value=dataset[2], name="b0")
    c0 = theano.shared(value=dataset[3], name="c0")
    k = T.vector('k')
    a_t = np.e ** (-(k[0] + k[1]) * t0)
    b_t = k[0] / (k[0] + k[1]) * (1 - a_t)
    c_t = k[1] / (k[0] + k[1]) * (1 - a_t)
    f = T.sum((a0 - a_t) ** 2 + (b0 - b_t) ** 2 + (c0 - c_t) ** 2)
    F = theano.function([k], f)
    g_f_k = T.jacobian(f, k)
    j_f_k = theano.function([k], g_f_k)
    H_f_k = T.hessian(f, k)
    Hessian = theano.function([k], H_f_k)

    maxiter = 1000
    track, f_val = [], []
    track.append(array(x))
    f_val.append(F(x))
    g = j_f_k(x)
    i = 0
    print "step =", i, "g=", g, "x=", x, "loss=", F(x),

    while (g ** 2).sum() > 10e-18:
        i += 1
        if i > maxiter:
            break
        G = Hessian(x)
        s = -np.linalg.solve(G, g)
        x += s
        print "step =", i, "g=", g, "x=", x, "loss=", F(x), "G=", G
        track.append(array(x))
        f_val.append(F(x))
        g = j_f_k(x)

    return x, F(x), track, f_val

def newton(func, grad_func, hessian_func, x0, dataset, linear=False, maxiter=1000):

    x = np.array(x0, dtype=np.float32)
    g = grad_func(x, dataset)

    track, f_val = [], []

    track.append(array(x))
    f_val.append(func(x, dataset))
    i = 0
    print "step =", i, "g=", g, "x=", x, "loss=", func(x, dataset)
    while (g ** 2).sum() > 10e-18:
        i += 1
        if i > maxiter:
            break
        G = hessian_func(x, dataset)
        s = np.linalg.solve(G, -g)
        if linear:
            alpha = backtracking_armijo(x, s, func, grad_func, dataset)
        else:
            alpha = 1
        x += alpha * s
        print "step =", i, "g=", g, "x=", x, "loss=", func(x, dataset), "G=", G
        track.append(array(x))
        f_val.append(func(x, dataset))
        g = grad_func(x, dataset)
    return x, func(x, dataset), track, f_val


def plot(func, dataset, x1s, x1e, x2s, x2e, delta, levels, track=None, f_val=None):
    # t, a, b, c = dataset

    k1 = np.arange(x1s, x1e, delta)
    k2 = np.arange(x2s, x2e, delta)
    K1, K2 = np.meshgrid(k1, k2)
    FX = np.zeros(K1.shape)
    row, col = K1.shape
    for i in xrange(row):
        for j in xrange(col):
            FX[i, j] = func(array([K1[i, j], K2[i, j]]), dataset)

    fig = plt.figure()
    subfig1 = fig.add_subplot(1, 2, 1)
    surf1 = plt.contour(K1, K2, FX, levels=levels, stride=0.001)

    if track:
        track = array(track)
        plt.plot(track[:, 0], track[:, 1])
    plt.xlabel("k1")
    plt.ylabel("k2")

    subfig2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = subfig2.plot_wireframe(K1, K2, FX, rstride=10, cstride=10, color='y')
    plt.contour(K1, K2, FX, stride=1, levels=levels)
    if track != None and f_val != None:
        f_val = array(f_val)
        subfig2.scatter(track[:, 0], track[:, 1], f_val)
    plt.show()


def problem1():
    def reactor(x, t):
        k1 = x[0]
        k2 = x[1]
        a_t = np.e ** (-(k1 + k2) * t)
        if k1+k2 == 0:
            b_t, c_t = np.zeros(a_t.shape), np.zeros(a_t.shape)
        else:
            b_t = k1 * 1.0 / (k1 + k2) * (1 - a_t)
            c_t = k2 * 1.0 / (k1 + k2) * (1 - a_t)
        return a_t, b_t, c_t

    def loss(x, dataset):
        t, a, b, c = dataset
        k1 = x[0]
        k2 = x[1]
        a_t, b_t, c_t = reactor(x, t)
        return sum((a-a_t) ** 2 + (b - b_t) ** 2 + (c-c_t) ** 2)

    def gradient(x, dataset):
        t, a, b, c = dataset
        k1 = x[0] * 1.
        k2 = x[1] * 1.
        a_t, b_t, c_t = reactor(x, t)
        gfa = 2. * (a_t - a)
        gfb = 2. * (b_t - b)
        gfc = 2. * (c_t - c)
        # JyF = array([gfa, gfb, gfc])
        # print "gfa", gfa
        g_a_k1 = -t * a_t
        g_b_k1 = 1. / (k1 + k2) * c_t + k1 / (k1 + k2) * t * a_t
        g_c_k1 = -1. / (k1 + k2) * c_t + k2 / (k1 + k2) * t * a_t
        g_a_k2 = -t * a_t
        g_b_k2 = -1. / (k1 + k2) * b_t + k1 / (k1 + k2) * t * a_t
        g_c_k2 = 1. / (k1 + k2) * b_t + k2 / (k1 + k2) * t * a_t
        JkY = array([[g_a_k1, g_a_k2],
                     [g_b_k1, g_b_k2],
                     [g_c_k1, g_c_k2]])
        # print "JkY", JkY
        gf = np.hstack([gfa, gfb, gfc])
        abc_k1 = np.hstack([g_a_k1, g_b_k1, g_c_k1])
        abc_k2 = np.hstack([g_a_k2, g_b_k2, g_c_k2])
        # n = t.shape[0]
        # JkF = np.zeros(x.shape)

        # for i in range(n):
        #     JkF += dot(JyF[:, i], JkY[:, :, i])
        # print "JkF", JkF
        return np.array([np.dot(gf, abc_k1), np.dot(gf, abc_k2)])

    def hessian(x, dataset):
        t0, a0, b0, c0 = dataset
        k1 = x[0]
        k2 = x[1]
        a_t0, b_t0, c_t0 = reactor(x, t0)
        n = t0.shape[0]
        HkF = np.zeros((2, 2))
        for i in range(n):
            t, a, b, c = t0[i], a0[i], b0[i], c0[i]
            a_t, b_t, c_t = a_t0[i], b_t0[i], c_t0[i]
            g_a_k1 = -t * a_t
            g_b_k1 = 1. / (k1 + k2) * c_t + k1 / (k1 + k2) * t * a_t
            g_c_k1 = -1. / (k1 + k2) * c_t + k2 / (k1 + k2) * t * a_t
            g_a_k2 = -t * a_t
            g_b_k2 = -1. / (k1 + k2) * b_t + k1 / (k1 + k2) * t * a_t
            g_c_k2 = 1. / (k1 + k2) * b_t + k2 / (k1 + k2) * t * a_t
            JkY = np.array([[g_a_k1, g_a_k2],
                            [g_b_k1, g_b_k2],
                            [g_c_k1, g_c_k2]])
            g_f_a = 2 * (a_t - a)
            g_f_b = 2 * (b_t - b)
            g_f_c = 2 * (c_t - c)
            JyF = np.array([g_f_a, g_f_b, g_f_c])
            a2_k1k1 = -t*g_a_k1
            b2_k1k1 = -1./((k1+k2)**2)*c_t + 1./(k1+k2)*g_c_k1 + k2/((k1+k2)**2)*t*a_t + k1/(k1+k2)*t*g_a_k1
            c2_k1k1 = 1./((k1+k2)**2)*c_t - 1./(k1+k2)*g_c_k1 - k2/((k1+k2)**2)*t*a_t + k2/(k1 + k2)*t*g_a_k1
            a2_k1k2 = -t*g_a_k2
            b2_k1k2 = -1./((k1+k2)**2)*c_t + 1./(k1+k2)*g_c_k2 - k1/((k1+k2)**2)*t*a_t + k1/(k1+k2)*t*g_a_k2  #checked
            c2_k1k2 = 1./((k1+k2)**2)*c_t - 1./(k1+k2)*g_c_k2 + k1/((k1+k2)**2)*t*a_t + k2/(k1+k2)*t*g_a_k2
            a2_k2k1 = -t*g_a_k1
            b2_k2k1 = 1./((k1+k2)**2)*b_t - 1./(k1+k2)*g_b_k1 + k2/((k1+k2)**2)*t*a_t + k1/(k1+k2)*t*g_a_k1   #checked
            c2_k2k1 = -1./((k1+k2)**2)*b_t + 1./(k1+k2)*g_b_k1 - k2/((k1+k2)**2)*t*a_t + k2/(k1+k2)*t*g_a_k1
            a2_k2k2 = -t*g_a_k2
            b2_k2k2 = 1./((k1+k2)**2)*b_t - 1./(k1+k2)*g_b_k2 - k1/((k1+k2)**2)*t*a_t + k1/(k1+k2)*t*g_a_k2
            c2_k2k2 = -1./((k1+k2)**2)*b_t + 1./(k1+k2)*g_b_k2 + k1/((k1+k2)**2)*t*a_t + k2/(k1+k2)*t*g_a_k2
            HkY = np.array([[[a2_k1k1, b2_k1k1, c2_k1k1], [a2_k1k2, b2_k1k2, c2_k1k2]],
                            [[a2_k2k1, b2_k2k1, c2_k2k1], [a2_k2k2, b2_k2k2, c2_k2k2]]])
            HyF = np.eye(3) * 2
            HkF += dot(dot(JkY.T, HyF), JkY) + dot(HkY, JyF)
        return HkF

    dataset = loadData("data/data.csv")
    plot(loss, dataset, 0, 1.2, 0, 1, 0.01, np.arange(0, 1, 0.1))
    x, fmin, track, f_val = newton(loss, gradient, hessian, [0.5, 0.4], dataset)
    plot(loss, dataset, 0.4, 0.6, 0.3, 0.5, 0.001, np.arange(0, 1, 0.01), track=track, f_val=f_val)

    x, fmin, track, f_val = newton(loss, gradient, hessian, [2, 2], dataset)
    plot(loss, dataset, 0, 120, 1, 110, 1, np.arange(0, 7, 0.1), track=track, f_val=f_val)
    # x, fmin, track, f_val = newton(loss, gradient, hessian, [2, 2], dataset, linear=True)
    # plot(loss, dataset, 0, 5, 0, 5, 1, np.arange(0, 3, 0.1), track=track, f_val=f_val)
    # x, fmin, track = newton(loss, gradient, hessian, [2, 2], dataset)
    # plot(loss, dataset, track=track)

    # x, fmin, track, f_val = auto4check(dataset, [0.5, 0.4])
    # plot(loss, dataset, 0, 1.2, 0, 1, 0.01, np.arange(0, 1, 0.1), track, f_val)
    x, fmin, track, f_val = auto4check(dataset, [2, 2])
    plot(loss, dataset, 0, 120, 1, 110, 1, np.arange(0, 7, 0.1), track, f_val)
if __name__ == "__main__":
    problem1()