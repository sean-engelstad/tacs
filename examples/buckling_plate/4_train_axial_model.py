import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import niceplots

"""
This time I'll try a Gaussian Process model to fit the axial critical load surrogate model
Inputs: D*, a0/b0, ln(b/h)
Output: k_x0
"""

# load the Nxcrit dataset
df = pd.read_csv("data/Nxcrit.csv")

# extract only the model columns
# TODO : if need more inputs => could maybe try adding log(E11/E22) in as a parameter?
# or also log(E11/G12)
X = df[["Dstar", "a0/b0", "b/h"]].to_numpy()
Y = df["kx_0"].to_numpy()
Y = np.reshape(Y, newshape=(Y.shape[0],1))

# convert b/h column to ln(b/h)
X[:,2] = np.log(X[:,2])

# split into training and test datasets
n_train = 1800
n_total = X.shape[0]
n_test = n_total - n_train

X_train = X[:n_train,:]
X_test = X[n_train:,:]
Y_train = Y[:n_train,:]
Y_test = Y[n_train:,:]

print(f"Y train = {Y_train.shape}")

# set model hyperparameters
sigma_n = 1e-2
sigma_f = 2.0
char_length = 1.0

# choose a Gaussian kernel
def kernel(xp, xq):
    # xp, xq are Nx1,Mx1 vectors (D*, a0/b0, ln(b/h))
    return sigma_f**2 * np.exp(-0.5/char_length**2 * (xp-xq) @ (xp-xq).T) 

# compute the base matrices
K = np.array([[kernel(X_train[i,:],X_train[j,:]) for i in range(n_train)] for j in range(n_train)]) + sigma_n**2 * np.eye(n_train)

# predicted mean and covariance are for the train/test set
# f* = K(X*,X) * (K(X,X) + sn^2*I)^{-1} * Y
# cov(f*) = K(X*,X*) - K(X*,X)^T * (K(X,X) + sn^2*I)^-1 * K(X,X*)

# predict against the test dataset
# --------------------------------------------------------------------------------------------------
# K(X*,X)
Kstar = np.array([[kernel(X_train[i,:],X_test[j,:]) for i in range(n_train)] for j in range(n_test)])

basis = np.linalg.solve(K, Y_train)
f_test = Kstar @ basis
# compute the RMSE on the test dataset
test_resid = Y_test - f_test
print(f"test resid = {test_resid}")
RMSE = 1.0/n_test * float(test_resid.T @ test_resid)
print(f"RMSE test = {RMSE}")

# plot the model and some of the data near the model range in D*=1, AR from 0.5 to 5.0, b/h=100
# ---------------------------------------------------------------------------------------------
X_plot = np.zeros((50,3))
X_plot[:,0] = 1.0
X_plot[:,1] = np.linspace(0.5, 10.0, 50)
X_plot[:,2] = np.log(75.0)

Kplot = np.array([[kernel(X_train[i,:],X_plot[j,:]) for i in range(n_train)] for j in range(50)])
f_plot = Kplot @ basis

# plot data in certain range of the training set
mask = np.logical_and(X[:,0] > 0.95, X[:,2] >= np.log(50.0), X[:,2] <= 100.0)
X_in_range = X[mask,:]
Y_in_range = Y[mask,:]

AR_in_range = X_in_range[:,1]

# plot the raw data and the model in this range
plt.style.use(niceplots.get_style())
plt.figure("check model")
plt.plot(X_plot[:,1], f_plot[:,0], label="pred")
plt.plot(AR_in_range,Y_in_range, "o", label="raw-data")
plt.xlabel("affine-AR")
plt.ylabel("kx_0")
plt.legend()
plt.savefig("data/isotropic-ml-pred.png", dpi=400)