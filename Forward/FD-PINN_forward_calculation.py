import deepxde as dde
import numpy as np
import math
import shutil
import tensorflow as tf
import os
from deepxde import utils
from deepxde.icbc.boundary_conditions import npfunc_range_autocache
import time
from pyDOE import lhs
import matplotlib
import matplotlib.pyplot as plt
import pickle
import scipy.io

import os

PI = math.pi
m=1
ks=0.025
EI=500
v=1

L=10
p0=20
w=(PI**2/L**2)*(EI/m)**0.5
fmax=0.5

dde.config.real.set_float64()
dde.config.set_default_float("float64")

geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, fmax)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

xx = np.linspace(0, L, 21)
ww = np.linspace(0, fmax, 21)
xx, ww = np.meshgrid(xx, ww)

xx = xx.flatten()[:, None]
ww = ww.flatten()[:, None]

AN = np.concatenate((xx, ww), 1)

def pde(x, y):

    ur_xx=dde.grad.hessian(y, x,component=0, i=0, j=0)
    ur_xxxx=dde.grad.hessian(ur_xx, x, i=0, j=0)
    ui_xx=dde.grad.hessian(y, x,component=1, i=0, j=0)
    ui_xxxx=dde.grad.hessian(ui_xx, x, i=0, j=0)

    eq_ur=EI*ur_xxxx-2*ks*y[:,1:2]*(x[:, 1:2]*2*PI)**2-m*y[:,0:1]*(x[:, 1:2]*2*PI)**2-p0/v*(tf.cos(x[:, 0:1]*x[:, 1:2]*2*PI/v))
    eq_ui=EI*ui_xxxx+2*ks*y[:,0:1]*(x[:, 1:2]*2*PI)**2-m*y[:,1:2]*(x[:, 1:2]*2*PI)**2+p0/v*(tf.sin(x[:, 0:1]*x[:, 1:2]*2*PI/v))

    return [eq_ur,eq_ui]  #

def boundary_l(x, on_boundary):

    return on_boundary


def fun_ur_xx(x, y, _):
    ur_xx=dde.grad.hessian(y, x,component=0, i=0, j=0)
    return ur_xx

def fun_ui_xx(x, y, _):
    ui_xx=dde.grad.hessian(y, x,component=1, i=0, j=0)
    return ui_xx


bc1 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l,component=0)
bc2 = dde.icbc.DirichletBC(geomtime, lambda x: 0, boundary_l,component=1)
bc3 = dde.icbc.OperatorBC(geomtime, fun_ur_xx, boundary_l)
bc4 = dde.icbc.OperatorBC(geomtime, fun_ui_xx, boundary_l)


data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc1,bc2,bc3,bc4],
    num_domain=0,
    num_boundary=0,
    num_initial=0,
    anchors=AN
)

layer_size = [2] + [60] * 3 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.maps.FNN(
    layer_size, activation, initializer
)

def feature_transform(x):

    x0=x[:,0:1]/L
    x1=x[:,1:2]/fmax

    return tf.concat(
        [x0,x1], axis=1
    )
def output_transform(x, y):

    return tf.concat([y[:,0:1]*10,y[:,1:2]*10], axis=1  )



net.apply_feature_transform(feature_transform)

net.apply_output_transform(output_transform)

model = dde.Model(data, net)

dde.optimizers.set_LBFGS_options(ftol=0,gtol=1e-20,maxiter=400000)
model.compile("L-BFGS-B",loss_weights=[1,1,1e2,1e2,1e6,1e6])

# model.restore("./model/trained-model-41599.ckpt", verbose=1)
pde_residual_resampler = dde.callbacks.PDEPointResampler(period=100)
losshistory, train_state = model.train( epochs=3000, display_every=100,callbacks=[pde_residual_resampler],model_save_path="./model/trained-model")

loss_test = np.sum(losshistory.loss_test, axis=1)
np.savetxt('loss_test.txt', loss_test)

loss_train = np.sum(losshistory.loss_train, axis=1)
np.savetxt('loss_train.txt', loss_train)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

no=501
x_star = np.linspace(0, L, 101)
f_star = np.linspace(0, fmax, 41)

X1_star, F1_star = np.meshgrid(x_star,f_star)
F1_star = F1_star.flatten()[:, None]
X1_star = X1_star.flatten()[:, None]


X = np.vstack((np.ravel(X1_star), np.ravel(F1_star))).T

pred = model.predict(X)

ur_pred = pred[:, 0]
ui_pred = pred[:, 1]

np.savetxt('X1_star.txt', X1_star)
np.savetxt('F1_star.txt', F1_star)

np.savetxt('ur_pred.txt', ur_pred)
np.savetxt('ui_pred.txt', ui_pred)