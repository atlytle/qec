# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: env1
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports and definitions

# %%
import numpy as np


# %%
def dot2(A, B):
    "Matrix multiplication mod 2."
    return np.mod(np.dot(A,B), 2)

def basis_vec(size, index):
    "Basis vector in R^size, in the index direction."
    _v = np.zeros(size)
    _v[index] = 1.0
    return _v

def to_base10(binary_vec):
    "Convert binary vector to corresponding base 10 integer."
    res = 0
    for i, x in enumerate(binary_vec):
        res += x*(2**i)
    return int(res)



# %% [markdown]
# # Classical Hamming codes

# %% [markdown]
# Author: Andrew Lytle
# Reference: https://en.wikipedia.org/wiki/Hamming(7,4)  
# The [7, 4] code encodes 4 logical bits into 7 encoded bits, using the additional bits as parity checks to detect all single-bit and two-bit errors, and can correct single-bit errors. (The minimal Hamming distance between any two codewords is 3). There is a very nice image that encapsulates how this works:  
# <img src="./Hamming7_4.svg" alt="Graphical depiction of Hamming [7,4,3]" />  
# Here the 'd' entries are the encoded logical bits (data), the 'p' entries give the parity checks on the data.
#
# Let us think for a moment about how the encoding will work.The parity bits should identify which bit has been corrupted---we will take bits 1, 2, and 4 as parity bits, and reading the value of these will indicate which of the 7 bits has been corrupted, with 000 indicating no error! In general, if we have $m$ parity bits, we can encode $2^m-m-1$ logical bits with the parity bits giving the syndrome measurement. This structure gives a class of [$2^m-1$, $2^m-m-1$, 3] codes.
#
# To generate the code words, we need a $7 \times 4$ matrix. Reading off the image,
# we encode the parity entries according to coverage of the data entries.
# For example, p1 covers d1, d2, and d4, so it corresponds to the row [1, 1, 0, 1].
# The data simply "passes through" i.e. the 'd' entries form a $4 \times 4$ identity matrix.

# %%
G = np.array([
              [1, 1, 0, 1],  # p1     
              [1, 0, 1, 1],  # p2
              [1, 0, 0, 0],  # d1
              [0, 1, 1, 1],  # p4
              [0, 1, 0, 0],  # d2
              [0, 0, 1, 0],  # d3
              [0, 0, 0, 1]   # d4
              ]) 

# %% [markdown]
# The related matrix $H$ is the parity check matrix, used to compute the error syndrome on a code word. By construction, we want $Hx=0$, $\forall x \in C$. $H$ encodes the parity check equations, and so is $3 \times 7$ (in general $(n-k) \times n$). Again referring to the figure

# %%
H = np.array([
            [1, 0, 1, 0, 1, 0, 1],  # p1 = d1 + d2 + d4 
            [0, 1, 1, 0, 0, 1, 1],  # p2 = d1 + d3 + d4
            [0, 0, 0, 1, 1, 1, 1]  # p3 = d2 + d3 + d4 
            ])

# %% [markdown]
# By construction, $G$ will take us from a 4-bit vector to a 7-bit encoded vector, which will have a parity check of 0, since it is a valid codeword. In other words, $H G = 0$. If there is a single bit-flip error, $H$ will tell us where it occurred.

# %%
dot2(H, G)

# %% [markdown]
# We can see below how $H$ can be used to detect, and correct an error.
# We also introduce a decoding matrix $R$ that simply pulls out the data bits.
# Note that an error on a parity bit, while detected, doesn't need to be corrected
# in order for the word to be correctly transmitted.

# %%
x_l = np.array([1,1, 0, 0])  # Logical word.
print(f"{x_l = }")
x_p = dot2(G, x)  # Physical codeword.
print(f"{x_p = }")
print(f"{dot2(H,x_p) = }")
x_p_e = x_p + basis_vec(7, 2)  # Introduce an error on bit 3.
syndrome = dot2(H, x_p_e)
print(to_base10(syndrome))  # Error occurred on this bit.
x_p_c = x_p_e + basis_vec(7, to_base10(syndrome)-1)
print(dot2(H, x_p_c))
R = np.array([
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
])
x_l_t = dot2(R, x_p_c)  # Transmitted logical word.
print(f"{x_l_t = }")
print(f"Word transmitted succesfully: {(x_l_t == x_l).all()}")


# %%
def test_single_bitflip():
    rng = np.random.default_rng()
    def random_word():

# %% [markdown]
# # Dual codes

# %% [markdown]
# For any linear code $C$ (recall that a linear code is defined by the property that x+y is a codeword if x and y are codewords), we may define a dual code $C^\perp$ as the set of all vectors $x$ with $<x, c>=0$, $\forall c \in C$. This inner product is mod 2, and so a given vector $x$ can be in both $C$ and $C^\perp$ (i.e. our "usual" intuition about orthogonality for vector spaces in $R^n$ doesn't hold here).
#
# The generator matrix of the $C$ is the parity check matrix of $C^\perp$, and vice versa.
# Let's consider a concrete example of this.

# %% [markdown]
# # CSS codes

# %% [markdown]
#
