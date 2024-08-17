# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports and definitions

# %%
import numpy as np


# %%
def to_diag(S, m, n):
    "m x n matrix with (m) elements of S along diagonal, padded with zeros."
    res = np.zeros((m,n))
    for i, val in enumerate(S):
        res[i,i] = val
    return res

def dotall(Ms):
    "Multiply matrices [M1, M2, ..., Mn] together."
    res = np.identity(Ms[-1].shape[1])
    for M in reversed(Ms):
        res = np.dot(M, res)
    return res



# %% [markdown]
# # Singular value decomposition

# %% [markdown]
# The singular value decomposition is an extremely general matrix factorization, which exists for any $m \times n$ complex matrix $M$.
# We write $M = U_{m \times m} \Sigma_{m \times n} V^\dagger_{n \times n}$, where $U$ and $V$ are *unitary*, and $\Sigma$ has non-negative real numbers along the diagonal. The size of the matrices has been written out in the subscripts. The elements of $\Sigma$ (called singular values) are uniquely determined, and the number of non-zero elements gives the *rank* of $M$. We can take these elements to be in descending order, say, which then makes $\Sigma$ uniquely determined by $M$.
#
# Recall that the *rank* of a matrix is the dimension of the vector space spanned by its columns. $r \leq \min{(m,n)}$. We can take $\Sigma$ to be $r \times r$ as well (remove zero entries), in which case $M = U_{m \times r} \Sigma_{r \times r} V^\dagger_{r \times n}$ and the columns/rows of $U$/$V^\dagger$ corresponding to zero entries have been removed.

# %%
m, n = 9, 6  # Matrix dimensions.
M = np.random.randn(m, n) + 1j*np.random.randn(m, n)

# Full  SVD.
U, S, Vh = np.linalg.svd(M, full_matrices=True)
print(U.shape, S.shape, Vh.shape)  # S is a list of singular values.
S = to_diag(S, m, n)  # Convert S to a diagonal matrix.
np.allclose(M, dotall([U, S, Vh]))

# Reduced SVD.
U, S, Vh = np.linalg.svd(M, full_matrices=False)
print(U.shape, S.shape, Vh.shape)  # S is a list of singular values.
r = S.shape[0]
S = to_diag(S, r, r)  # Convert S to a diagonal matrix.
np.allclose(M, dotall([U, S, Vh]))


# %% [markdown]
# # Schmidt decomposition and entanglement

# %% [markdown]
# The Schmidt decomposition of a vector is essentially a rephrasing of SVD. It can tell us e.g. about entanglement in a quantum state.

# %% [markdown]
#
