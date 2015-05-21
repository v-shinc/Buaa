__author__ = 'chensn'

import numpy as np
from numpy import dot, array, eye, outer, sqrt
from numpy.linalg import norm
import csv
import theano
import theano.tensor as T


# Backtracking-Armijo line search
def backtracking_armijo(x0, p, func, grad_func, *dataset, **kwargs):
    alpha = 1
    rho = kwargs.pop("rho", 0.25)
    gamma = kwargs.pop("gamma", 0.1)
    while not (func(x0 + alpha * p, *dataset)) <= func(x0, *dataset) + rho * np.dot(grad_func(x0, *dataset), p) * alpha:
        alpha *= gamma
    return alpha


def bfgs(func, grad_func, x0, dataset, tol=1e-9, maxiter=1000):
    x = array(x0, dtype=np.float64)
    g = grad_func(x, *dataset)
    track, f_val = [], []
    H = eye(x.shape[0])
    I = eye(x.shape[0])
    track.append(array(x))
    f_val.append(func(x, *dataset))
    i = 0
    print "Step =", i, "g=", g, "x=", x, "loss=", func(x, *dataset), "H=", H
    while norm(g) > tol:
        i += 1
        if i > maxiter:
            break
        p = -dot(H, g)
        alpha = backtracking_armijo(x, p, func, grad_func, *dataset, rho=0.25, gamma=0.78)
        x += alpha * p
        track.append(array(x))
        f_val.append(func(x, *dataset))
        s = alpha * p
        g1 = grad_func(x, *dataset)
        y = g1 - g
        rho = 1. / dot(y, s)
        if rho == np.inf:
            print "rho is inf"
            break
        H = dot((I - rho * outer(s, y)), dot(H, (I - rho * outer(y, s)))) + rho * outer(s, s)
        g = g1
        print "step =", i, "g=", g, "x=", x, "loss=", func(x, *dataset), "H=", H
    return x, func(x, *dataset), track, f_val


class TrustRegion():
    def __init__(self, func, grad_func, hessian_func, delta, args, tol=1e-9, maxiter=100):
        self.func = func
        self.grad = grad_func
        self.hessian = hessian_func
        self.args = args
        self.delta = delta
        self.tol = tol
        self.maxiter = maxiter

    def get_tau(self, x, p):
        """Function to find tau such that ||x + tau.p|| = delta."""
        p_x = dot(p, x)
        p_2 = dot(p, p)
        tau = (-p_x + sqrt(p_x ** 2 - p_2 * (dot(x, x) - self.delta ** 2))) / p_2
        return tau

    def steihaug(self, x, g, B, epsilon=1e-8):
        _x, r, p = np.zeros_like(x), g, -g
        if norm(r) < epsilon:
            return _x
        while 1:
            curv = dot(p, dot(B, p))
            # B is not positive
            if curv <= 0:
                return _x + self.get_tau(_x, p) * p
            alpha = dot(r, r) / curv
            _x += alpha * p
            if norm(_x) >= self.delta:
                return _x + self.get_tau(_x, p) * p
            r1 = r + alpha * dot(B, p)
            if norm(r1) < epsilon * norm(r):
                return _x
            beta = dot(r1, r1) / dot(r, r)
            p = -r1 + beta * p
            r = r1

    def solve(self, x0):
        track, f_val = [], []
        i = 0
        x = array(x0, dtype=np.float64)
        track.append(array(x))
        f_val.append(self.func(x, *self.args))
        print "step =", i, "g=", self.grad(x, *self.args), "x=", x, "loss=", self.func(x, *self.args)
        i += 1
        while 1:
            g = self.grad(x, *self.args)
            G = self.hessian(x, *self.args)
            s = self.steihaug(x, g, G)
            f_k = self.func(x, *self.args)
            q_k = f_k + dot(g, s) + 0.5 * dot(s, dot(G, s))
            delta_f = f_k - self.func(x + s, *self.args)
            delta_q = f_k - q_k
            rho = delta_f / delta_q
            if rho < 0.25:
                self.delta = norm(s) / 4
            if rho > 0.75 and norm(s) == self.delta:
                self.delta *= 2
            if rho <= 0:
                pass
            else:
                x += s
            if norm(g) < self.tol:
                break
            track.append(array(x))
            f_val.append(self.func(x, *self.args))
            print "step =", i, "g=", g, "x=", x, "loss=", self.func(x, *self.args)
            i += 1
            if i > self.maxiter:
                break
        return x, self.func(x, *self.args), track, f_val


def newton(func, grad_func, hessian_func, x0, dataset, tol=1e-9, linear=False, maxiter=1000):

    x = np.array(x0, dtype=np.float64)
    g = grad_func(x, *dataset)

    track, f_val = [], []

    track.append(array(x))
    f_val.append(func(x, *dataset))
    i = 0
    print "step =", i, "g=", g, "x=", x, "loss=", func(x, *dataset)
    while norm(g) > tol:
        i += 1
        if i > maxiter:
            break
        G = hessian_func(x, *dataset)
        s = np.linalg.solve(G, -g)
        if linear:
            alpha = backtracking_armijo(x, s, func, grad_func, *dataset)
        else:
            alpha = 1
        x += alpha * s
        track.append(array(x))
        f_val.append(func(x, *dataset))
        g = grad_func(x, *dataset)
        print "step =", i, "g=", g, "x=", x, "loss=", func(x, *dataset), "G=", G
    return x, func(x, *dataset), track, f_val








