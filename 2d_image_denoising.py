import copy
import numpy as np
import matplotlib.pyplot as plt

#Variable Definition
alpha = 0.1
lam = 100
gamma = 0.3
tau = 0.99/8 #0.99 / 8
mu = 0.1
width = 20
height = 20

np.random.seed(1)

# x：二次元 -> D(x)：三次元
def D(x):
    Dx = np.zeros((height, width, 2))
    Dx[:-1,:,0] = x[1:,:] - x[:-1,:]
    Dx[:,:-1,1] = x[:,1:] - x[:,:-1]
    return Dx

# u：三次元 -> trans_D(u)：二次元
def trans_D(u):
    tDu = np.zeros((height, width))
    tDu[:,:] = -u[:,:,0]
    tDu[1:,:] += u[:-1,:,0]
    tDu[:,:] += -u[:,:,1]
    tDu[:,1:] += u[:,:-1,1]
    return tDu

# v：二次元 * 3　-> u：三次元
def C(v1, v2, v3):
    C_v1v2v3 = np.zeros((height, width, 2))
    tmp = np.zeros((height,width))
    C_v1v2v3[:,:,0] = v1[:,:,0]
    tmp[:,:] = v2[:,:,0]
    tmp[:-1,1:] += v2[1:,:-1,0]
    tmp[:-1,:] += v2[1:,:,0]
    tmp[:,1:] += v2[:,:-1,0]
    C_v1v2v3[:,:,0] += (tmp * 0.25)
    tmp[:,:] = v3[:,:,0]
    tmp[:-1,:] += v3[1:,:,0]
    C_v1v2v3[:,:,0] += (tmp * 0.5)

    tmp[:,:] = v1[:,:,1]
    tmp[1:,:-1] += v1[:-1,1:,1]
    tmp[1:,:] += v1[:-1,:,1]
    tmp[:,:-1] += v1[:,1:,1]
    C_v1v2v3[:,:,1] = (tmp * 0.25)
    C_v1v2v3[:,:,1] += v2[:,:,1]
    tmp[:,:] = v3[:,:,1]
    tmp[:,:-1] = v3[:,1:,1]
    C_v1v2v3[:,:,1] += (tmp * 0.5)

    return -C_v1v2v3

# u：三次元 -> v：二次元 * 3
def trans_C(u):
    v1 = np.zeros((height, width, 2))
    v2 = np.zeros((height, width, 2))
    v3 = np.zeros((height, width, 2))


    v1[:,:,0] = u[:,:,0]
    v1[:,:,1] = u[:,:,1]
    v1[:-1,1:,1] += u[1:,:-1,1]
    v1[:-1,:,1] += u[1:,:,1]
    v1[:,:-1,1] += u[:,1:,1]
    v1[:,:,1] *= 0.25

    v2[:,:,0] = u[:,:,0]
    v2[1:,:-1,0] += u[:-1,1:,0]
    v2[1:,:,0] += u[:-1,:,0]
    v2[:,:-1,0] += u[:,1:,0]
    v2[:,:,0] *= 0.25
    v2[:,:,1] = u[:,:,1]

    v3[:,:,0] = u[:,:,0]
    v3[1:,:,0] = u[:-1,:,0]
    v3[:,:,0] *= 0.5
    v3[:,:,1] = u[:,:,1]
    v3[:,1:,1] += u[:,:-1,1]
    v3[:,:,1] *= 0.5
    
    return -v1, -v2, -v3


def proxF(x, y):
    return (x + alpha * y) / (1 + alpha)

# v↕︎ or v↔︎ or v.
def proxG(v):
    proxGv = np.zeros((height, width, 2))
    proxGv[:,:,0] = v[:,:,0] - (v[:,:,0] / np.maximum(np.abs(v[:,:,0] / (alpha * lam)), 1))
    proxGv[:,:,1] = v[:,:,1] - (v[:,:,1] / np.maximum(np.abs(v[:,:,1] / (alpha * lam)), 1))
    return proxGv
    

#Noisy image definition y[height, width]
y = np.zeros((height, width), dtype=int)
for i in range(height):
    for k in range(width):
        if i + k < (height + width) / 2:
            y[i][k] = 256
        tmp = np.random.randint(1, 100)
        if tmp < 30:
            y[i][k] = np.random.randint(0, 256)

x = np.zeros((height, width))
u = np.zeros((height, width, 2))
v1 = np.zeros((height, width, 2))
v2 = np.zeros((height, width, 2))
v3 = np.zeros((height, width, 2))

for n1 in range(height):
    for n2 in range(width):
        x[n1, n2] = np.random.randint(1, 10)
        v1[n1, n2] = np.random.randint(1, 10)
        v2[n1, n2] = np.random.randint(1, 10)
        v3[n1, n2] = np.random.randint(1, 10)
        for i in range(2):
            u[n1, n2, i] = np.random.randint(1, 10)



diff = 10
count = 0

while diff > 1e-5:
    old_u = u
    count +=1
    x = proxF(x - tau*trans_D(D(x) + C(v1, v2, v3) + mu*u), y)

    t1, t2, t3 = trans_C(D(x) + C(v1, v2, v3) + mu*u)
    v1 = proxG(v1 - gamma*t1)
    v2 = proxG(v2 - gamma*t2)
    v3 = proxG(v3 - gamma*t3)

    u = u + (D(x) + C(v1, v2, v3)) / mu

    diff = np.sum(np.abs(old_u - u))
    # if count > 500:
    #     break


np.savetxt("2d_ydata.txt", y)
np.savetxt("2d_xdata.txt", x)

print("count =", count)
print(np.sum(x - y))

plt.subplot(1, 2, 1)
plt.imshow(y, cmap='viridis')  # ヒートマップの色を指定
plt.colorbar()                    # カラーバーを表示
plt.title("y")

plt.subplot(1, 2, 2)
plt.imshow(x, cmap='viridis')  # ヒートマップの色を指定
plt.colorbar()                    # カラーバーを表示
plt.title("x")
plt.tight_layout()
plt.show()
