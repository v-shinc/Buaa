__author__ = 'chensn'

import numpy as np
from numpy import dot, array, eye, outer, sqrt
from numpy.linalg import norm
import csv
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import theano
import theano.tensor as T
from scipy.optimize import fmin_bfgs, fmin_l_bfgs_b
from optimize import TrustRegion, newton, bfgs

def loadData(file):
    dataset = []
    with open(file) as fin:
        reader = csv.reader(fin)
        for line in reader:
            dataset.append(line)
    dataset = np.array([[np.float64(d) for d in row] for row in dataset[1:]])
    t = dataset[:, 0]
    a = dataset[:, 1]
    b = dataset[:, 2]
    c = dataset[:, 3]
    return t, a, b, c




def auto4check(dataset, x, tol=1e-9, maxiter=1000):

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


    track, f_val = [], []
    track.append(array(x))
    f_val.append(F(x))
    g = j_f_k(x)
    i = 0
    print "Step =", i, "g=", g, "x=", x, "loss=", F(x)
    while norm(g) > tol:
        i += 1
        if i > maxiter:
            break
        G = Hessian(x)
        s = -np.linalg.solve(G, g)
        x += s
        track.append(array(x))
        f_val.append(F(x))
        g = j_f_k(x)
        print "step =", i, "g=", g, "x=", x, "loss=", F(x), "G=", G
    return x, F(x), track, f_val


def plot(func, dataset, x1s, x1e, x2s, x2e, delta, levels, title=None, track=None, f_val=None):
    # t, a, b, c = dataset

    k1 = np.arange(x1s, x1e, delta)
    k2 = np.arange(x2s, x2e, delta)
    K1, K2 = np.meshgrid(k1, k2)
    FX = np.zeros(K1.shape)
    row, col = K1.shape
    for i in xrange(row):
        for j in xrange(col):
            FX[i, j] = func(array([K1[i, j], K2[i, j]]), *dataset)

    fig = plt.figure(title)
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

    def loss(x, *dataset):
        t, a, b, c = dataset
        k1 = x[0]
        k2 = x[1]
        a_t, b_t, c_t = reactor(x, t)
        return sum((a-a_t) ** 2 + (b - b_t) ** 2 + (c-c_t) ** 2)

    def gradient(x, *dataset):
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

    def hessian(x, *dataset):
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
    # plot(loss, dataset, 0, 1.2, 0, 1, 0.01, np.arange(0, 1, 0.1))
    # x, fmin, track, f_val = newton(loss, gradient, hessian, [0.5, 0.4], dataset, tol=1e-17)
    # x, fmin, track, f_val = auto4check(dataset, [0.5, 0.4], tol=1e-17)
    # plot(loss, dataset, 0.4, 0.6, 0.3, 0.5, 0.001, np.arange(0, 1, 0.01), track=track, f_val=f_val)

    # x, fmin, track, f_val = newton(loss, gradient, hessian, [2, 2], dataset)
    # plot(loss, dataset, 0, 120, 1, 110, 1, np.arange(0, 7, 0.1), track=track, f_val=f_val)
    # x, fmin, track, f_val = newton(loss, gradient, hessian, [2, 2], dataset, linear=True)
    # plot(loss, dataset, 0, 5, 0, 5, 1, np.arange(0, 3, 0.1), track=track, f_val=f_val)
    # x, fmin, track = newton(loss, gradient, hessian, [2, 2], dataset)
    # plot(loss, dataset, track=track)

    # x, fmin, track, f_val = auto4check(dataset, [0.5, 0.4])
    # plot(loss, dataset, 0, 1.2, 0, 1, 0.01, np.arange(0, 1, 0.1), track, f_val)
    # x, fmin, track, f_val = auto4check(dataset, [2, 2])
    # plot(loss, dataset, 0, 120, 1, 110, 1, np.arange(0, 7, 0.1), track, f_val)
    print fmin_l_bfgs_b(loss, [2, 2], fprime=gradient, args=dataset, approx_grad=False, bounds=None, epsilon=1e-16, iprint=0, disp=1)


def auto4check2(input,dataset):
    a = theano.shared(value=dataset[0], name="a")
    b = theano.shared(value=dataset[1], name="b")
    c = theano.shared(value=dataset[2], name="c")
    x = T.vector('x')
    u = x[0] - 0.8
    v = x[1] - (a[0] + a[1] * u ** 2 * (1 - u) ** 0.5 - a[2] * u)
    alpha = -b[0] + b[1] * u ** 2 * (1 + u) ** 0.5 + b[2] * u
    beta = c[0] * v ** 2 * (1 - c[1] * v) / (1 + c[2] * u ** 2)
    fx = alpha * np.e ** (-beta)
    g_f_x = T.jacobian(fx, x)
    grad = theano.function([x], g_f_x)
    Hessian = theano.function([x], T.hessian(fx, x))
    H_alpha_x = theano.function([x], T.hessian(alpha, x))
    H_beta_x = theano.function([x], T.hessian(beta, x))
    J_f_alpha = theano.function([x], T.grad(fx, alpha))
    J_f_beta = theano.function([x], T.grad(fx, beta))
    J_alpha_x = theano.function([x], T.grad(alpha, x))

    J_beta_x = theano.function([x], T.grad(beta, x))

    J_f_y = [J_f_alpha(input), J_f_beta(input)]
    J_y_x = [J_alpha_x(input), J_beta_x(input)]
    # print "H_alpha_x"
    # print H_alpha_x(input)
    # print "H_beta_x"
    # print H_beta_x(input)
    # print "J_f_y"
    # print J_f_y
    # print "J_y_x"
    # print J_y_x
    # print grad(input)

    return Hessian(input)


def problem2():


    def component(x, *dataset):
        a, b, c = dataset
        u = x[0] - 0.8
        v = x[1] - (a[0] + a[1] * u ** 2 * (1 - u) ** 0.5 - a[2] * u)
        alpha = -b[0] + b[1] * u ** 2 * (1 + u) ** 0.5 + b[2] * u
        beta = c[0] * (v ** 2) * (1 - c[1] * v) / (1 + c[2] * (u ** 2))
        return u, v, alpha, beta
    def loss(x, *dataset):
        u, v, alpha, beta = component(x, *dataset)
        # a, b, c = dataset
        # u = x[0] - 0.8
        # v = x[1] - (a[0] + a[1] * u ** 2 * (1 - u) ** 0.5 - a[2] * u)
        # alpha = -b[0] + b[1] * u ** 2 * (1 + u) ** 0.5 + b[2] * u
        # beta = c[0] * (v ** 2) * (1 - c[1] * v) / (1 + c[2] * (u ** 2))
        return alpha * np.e ** (-beta)
    def gradient(x, *dataset):
        if x.dtype != np.float64:
            x = array(x, dtype=np.float64)
        a, b, c = dataset
        a1, a2, a3 = a
        b1, b2, b3 = b
        c1, c2, c3 = c
        u, v, alpha, beta = component(x, *dataset)
        g_v_u = -2*u*a2*((1-u)**0.5) + 0.5*a2*(u**2)*((1-u)**(-0.5)) + a3
        # g_v_x = array([g_v_u, 1])
        g_u_x = array([1., 0])
        g_beta_v = 1./(1+c3*u**2)*(2*c1*v-3*c1*c2*v**2)
        g_beta_u = -c1*(v**2)*(1-c2*v)*((1+c3*(u**2))**-2)*2*c3*u + g_beta_v * g_v_u
        g_alpha_u = 2*b2*u*(1+u)**0.5 + 0.5*b2*u**2*(1+u)**-0.5 + b3
        g_alpha_x = g_alpha_u * g_u_x
        # g_beta_x = g_beta_u*g_u_x + g_beta_v*g_v_x
        g_beta_x1 = g_beta_u * 1.
        g_beta_x2 = g_beta_v * 1.
        g_beta_x = [g_beta_x1, g_beta_x2]
        g_f_alpha = np.e ** -beta
        g_f_beta = -alpha * np.e ** -beta
        g_f_x = dot([g_f_alpha, g_f_beta], [g_alpha_x, g_beta_x])

        return g_f_x
    def min_eig(x, *dataset):
        H = hessian(x, *dataset)
        w, _ = np.linalg.eig(H)
        return min(w)
    def hessian(x, *dataset):
        # substitute x1 into u
        # y = [alpha, beta]
        # z = [u, v]
        x = array(x, dtype=np.float64)
        a, b, c = dataset
        a1, a2, a3 = a
        b1, b2, b3 = b
        c1, c2, c3 = c
        u, v, alpha, beta = component(x, *dataset)
        alpha_uu = 2*b2*(1+u)**0.5 + 2*b2*u*(1+u)**-0.5 - 0.25*b2*u**2*(1+u)**(-1.5)

        v_x1x1 = -2*a2*(1.8-x[0])**0.5 + 2*a2*(x[0]-0.8)*(1.8-x[0])**(-0.5) + 0.25*a2*(x[0]-0.8)**2*(1.8-x[0])**(-1.5)
        H_z_x = np.zeros((2, 2, 2))
        H_z_x[0, 0, 1] = v_x1x1

        beta_uu = -2*c1*c3*v**2*(1-c2*v)*(1-3*c3*u**2)/(1+c3*u**2)**3
        beta_uv = -2*c1*c3*u*(2*v-3*c2*v**2)/(1+c3*u**2)**2
        beta_vu = beta_uv
        beta_vv = c1/(1+c3*u**2)*(2-6*c2*v)

        H_alpha_z = array([[alpha_uu, 0],
                           [0, 0]])
        H_beta_z = array([[beta_uu, beta_uv],
                          [beta_vu, beta_vv]])
        u_x1 = 1
        u_x2 = 0
        v_x1 = -2*u*a2*((1-u)**0.5) + 0.5*a2*(u**2)*((1-u)**(-0.5)) + a3
        v_x2 = 1
        J_z_x = array([[u_x1, u_x2],
                       [v_x1, v_x2]])
        alpha_u = 2*b2*u*(1+u)**0.5 + 0.5*b2*u**2*(1+u)**-0.5 + b3

        alpha_v = 0
        beta_v = 1./(1+c3*u**2)*(2*c1*v-3*c1*c2*v**2)
        beta_u = -2*c1*c3*u*(v**2)*(1-c2*v)*((1+c3*(u**2))**-2)

        J_alpha_z = array([alpha_u, alpha_v])
        J_beta_z = array([beta_u, beta_v])
        H_alpha_x = dot(dot(J_z_x.T, H_alpha_z), J_z_x) + dot(H_z_x, J_alpha_z)
        H_beta_x = dot(dot(J_z_x.T, H_beta_z), J_z_x) + dot(H_z_x, J_beta_z)

        H_y_x = np.dstack((H_alpha_x, H_beta_x))
        H_f_y = array([[0, -np.e**(-beta)], [-np.e**(-beta), alpha*np.e**(-beta)]])
        alpha_x1 = alpha_u * 1
        alpha_x2 = 0
        beta_x1 = beta_u * 1. + beta_v * v_x1
        beta_x2 = beta_v * 1.
        J_f_y = array([np.e**(-beta), -alpha*np.e**(-beta)])

        J_y_x = array([[alpha_x1, alpha_x2], [beta_x1, beta_x2]])
        # print "J_f_y"
        # print J_f_y
        # print "J_y_x"
        # print J_y_x
        H_f_x = dot(dot(J_y_x.T, H_f_y), J_y_x) + dot(H_y_x, J_f_y)
        return H_f_x


    a = array([0.3, 0.6, 0.2], dtype=np.float64)
    b = array([5., 26., 3.], dtype=np.float64)
    c = array([40., 1., 10.], dtype=np.float64)
    dataset = (a, b, c)
    plot(loss, dataset, 0, 1, 0, 1, 0.01, np.arange(-10, 10, 0.5), title="contour")
    plot(min_eig, dataset, 0, 1, 0, 1, 0.01, np.arange(-260, 60, 8), title="Hessian eigenvalue contour")
    x, fmin, track, f_val = newton(loss, gradient, hessian, [0.8, 0.3], dataset, 1e-8)
    plot(loss, dataset, 0, 1, 0, 1, 0.01, np.arange(-10, 10, 0.5), track=track, f_val=f_val, title="basic newton (0.8, 0.3)")
    x, fmin, track, f_val = newton(loss, gradient, hessian, [1., 0.5], dataset, 1e-8)
    plot(loss, dataset, 0, 1.5, 0, 1, 0.01, np.arange(-10, 15, 0.5), track=track, f_val=f_val, title="basic newton (1, 0.5)")
    # print fmin_l_bfgs_b(loss, [0.8, 0.3], fprime=gradient, args=dataset, approx_grad=False, bounds=[(0.2, 1.8), (None, None)], epsilon=1e-8, iprint=0, disp=1)
    # print fmin_l_bfgs_b(loss, [1, 0.5], fprime=gradient, args=dataset, approx_grad=False, bounds=[(0.2, 1.8), (None, None)], epsilon=1e-8, iprint=0, disp=1)
    # print bfgs(loss, gradient, [0.8, 0.3], dataset, tol=1e-8)
    x, fmin, track, f_val = bfgs(loss, gradient, [1, 0.5], dataset, tol=1e-8)
    plot(loss, dataset, 0, 1.5, 0, 1, 0.01, np.arange(-10, 15, 0.5), track=track, f_val=f_val, title="bfgs (1, 0.5)")
    tr = TrustRegion(loss, gradient, hessian, delta=0.05, args=dataset, tol=1e-8)
    x, fmin, track, f_val = tr.solve([0.8, 0.3])
    plot(loss, dataset, 0, 1, 0, 1, 0.01, np.arange(-10, 10, 0.5), track=track, f_val=f_val, title="trust region cg (0.8, 0.3)")
    x, fmin, track, f_val = tr.solve([1.0, 0.5])
    plot(loss, dataset, 0, 1.5, 0, 1, 0.01, np.arange(-10, 15, 0.5), track=track, f_val=f_val, title="trust region cg (1, 0.5)")
if __name__ == "__main__":
    # problem1()
    problem2()


