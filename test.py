import torch
import numpy as np
N, D_in, H, D_out = 64, 1000, 100, 10
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)
learning_rate = 1e-6
for t in range(500):
    #forword pass
    h = x.dot(w1)
    h_relu = np.maximum(0, h)
    y_pre = h_relu.dot(w2)
    #compute loss
    loss = np.square(y_pre -y).sum()
    print(t, loss)
    #back

