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
    print(res)
    print(res.shape)
    for M in reversed(Ms):
        res = np.dot(M, res)
    return res



# %% [markdown]
# # Singular value decomposition

# %% [markdown]
# The singular value decomposition is an extremely general matrix factorization, which exists for any $m \times n$ complex matrix $M$.
# We write $M = U_{m \times m} \Sigma_{m \times n} V^\dagger_{n \times n}$, where $U$ and $V$ are *unitary*, and $\Sigma$ has non-negative real numbers along the diagonal. The size of the matrices has been written out in the subscripts. The elements of $\Sigma$ (called singular values) are uniquely determined and the number of non-zero elements gives the *rank* of $M$. We can take these elements to be in descending order, say, which then makes $\Sigma$ uniquely determined by $M$.
#
# Recall that the *rank* of a matrix is the dimension of the vector space spanned by its columns. The number of zero entries give the dimension of the *kernel*, or *null space*, i.e. the vector space that is mapped to 0 by $M$.

# %%
M = np.random.randn(9, 6) + 1j*np.random.randn(9, 6)
U, S, Vh = np.linalg.svd(M, full_matrices=True)
print(U.shape, S.shape, Vh.shape)
S = to_diag(S, 9, 6)
np.allclose(M, dotall([U, S, Vh]))


# %% [markdown]
# # Schmidt decomposition and entanglement

# %% [markdown]
#
