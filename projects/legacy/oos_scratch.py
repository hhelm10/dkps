import numpy as np
from graspologic.embed import ClassicalMDS
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.linear_model import LinearRegression
from scipy.linalg import inv

# np.random.seed(0)
dim_lo = 5
dim_hi = 20
n_obs  = 10

# sample low-dimensional points
x_lo = np.random.uniform(size=(n_obs, dim_lo))

# project into higher dimension
Q, _ = np.linalg.qr(np.random.randn(dim_hi, n_obs))
Q    = Q[:,:dim_lo]
x_hi = x_lo @ Q.T
assert np.allclose(squareform(pdist(x_lo)), squareform(pdist(x_hi))) # distances are preserved

# do MDS
d_hi = squareform(pdist(x_hi)) # / np.sqrt(x_hi.shape[0])
mds  = ClassicalMDS(n_components=2, dissimilarity='precomputed')
y_lo = mds.fit_transform(d_hi)

# attempt to reconstruct exactly
# mu         = x_lo.mean(axis=0, keepdims=True)
# x_lo_c     = x_lo - mu
# U, _, Vt   = np.linalg.svd(y_lo.T @ x_lo_c)
# R          = U @ Vt
# x_lo_rec   = y_lo @ R + mu
# print(f"MSE after rotation: {np.mean((x_lo_rec - x_lo)**2)}")

# --

def oos_mds(q_hi, y_lo):
    D2       = d_hi ** 2
    d2       = ((x_hi - q_hi) ** 2).sum(axis=1)
    b        = -0.5 * (d2 - D2.mean(axis=1) - d2.mean() + D2.mean())
    eigvals  = (y_lo ** 2).sum(axis=0)
    y_q      = (y_lo.T @ b) / eigvals
    # q_lo_rec = y_q @ R + mu
    # print("MSE(q_lo_rec, q_lo):", np.mean((q_lo_rec - q_lo)**2))
    return y_q


q_lo = np.random.uniform(0, 1, dim_lo)
q_hi = q_lo @ Q.T
y_q = oos_mds(q_hi, y_lo)

cdist(x_lo, [q_lo])
cdist(x_hi, [q_hi])
cdist(y_lo, [y_q])

squareform(pdist(y_lo))



# -------------------------


import numpy as np
from graspologic.embed import ClassicalMDS
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.linear_model import LinearRegression
from scipy.linalg import inv
from matplotlib import pyplot as plt

# np.random.seed(0)
dim_lo = 5
dim_hi = 20
n_obs  = 100
n_components = 2

# sample low-dimensional points
x_lo = np.random.uniform(size=(n_obs + 1, dim_lo))

# project into higher dimension
Q, _ = np.linalg.qr(np.random.randn(dim_hi, n_obs))
Q    = Q[:,:dim_lo]
x_hi = x_lo @ Q.T
assert np.allclose(squareform(pdist(x_lo)), squareform(pdist(x_hi))) # distances are preserved


# do MDS
d_hi = squareform(pdist(x_hi)) # / np.sqrt(x_hi.shape[0])
mds  = ClassicalMDS(n_components=n_components, dissimilarity='precomputed')
y_lo = mds.fit_transform(d_hi)

d_hi_q = squareform(pdist(x_hi[:-1])) # / np.sqrt(x_hi.shape[0])
mds    = ClassicalMDS(n_components=n_components, dissimilarity='precomputed')
y_lo_q = mds.fit_transform(d_hi_q)

# --

def oos_mds(x_hi, q_hi, y_lo):
    d_hi     = squareform(pdist(x_hi)) # / np.sqrt(x_hi.shape[0])
    D2       = d_hi ** 2
    d2       = ((x_hi - q_hi) ** 2).sum(axis=1)
    b        = -0.5 * (d2 - D2.mean(axis=1) - d2.mean() + D2.mean())
    eigvals  = (y_lo ** 2).sum(axis=0)
    y_q      = (y_lo.T @ b) / eigvals
    # q_lo_rec = y_q @ R + mu
    # print("MSE(q_lo_rec, q_lo):", np.mean((q_lo_rec - q_lo)**2))
    return y_q


y_q  = oos_mds(x_hi[:-1], x_hi[-1], y_lo_q)
y_q

a = cdist(y_lo[:-1], [y_lo[-1]]).squeeze()
b = cdist(y_lo_q, [y_q]).squeeze()
plt.scatter(a, b)
plt.show()


def oos_mds2(X_hi, X_lo, q_hi):
    D2       = squareform(pdist(X_hi, metric='sqeuclidean'))
    d2       = ((X_hi - q_hi) ** 2).sum(axis=1)
    b        = -0.5 * (d2 - D2.mean(axis=1) - d2.mean() + D2.mean())
    return (X_lo.T @ b) / (X_lo ** 2).sum(axis=0)

oos_mds2(x_hi[:-1], y_lo_q, x_hi[-1])


X_lo.T @ X_lo
(X_lo ** 2).sum(axis=0)

def oos_mds_least_squares(X_hi, X_lo, q_hi):
    D2 = squareform(pdist(X_hi, metric='sqeuclidean'))
    d2 = ((X_hi - q_hi) ** 2).sum(axis=1)
    b = -0.5 * (d2 - D2.mean(axis=1) - d2.mean() + D2.mean())
    
    XtX  = X_lo.T @ X_lo
    Xt_b = X_lo.T @ b
    return np.linalg.solve(XtX, Xt_b)

oos_mds2(x_hi[:-1], y_lo_q, x_hi[-1])
oos_mds_least_squares(x_hi[:-1], y_lo_q, x_hi[-1])


(I - M) D (I - M)

D - row_avgs - col_avgs + M D M

import numpy as np
n = 3
x = np.random.uniform(0, 1, (n, n))
d = squareform(pdist(x, metric='sqeuclidean'))

J = np.eye(n) - 1 / n * np.ones((n, n))
g = - 0.5 * J @ d @ J

z = x - x.mean(axis=0, keepdims=True)
np.allclose(z @ z.T, g)


