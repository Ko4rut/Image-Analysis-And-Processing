import numpy as np

def dft_1d(x):
    """
    DFT 1D (from scratch)
    x: array (N,)
    return: X array (N,) complex
    """
    x = np.asarray(x, dtype=np.complex128)
    N = x.size
    X = np.zeros(N, dtype=np.complex128)

    for m in range(N):  # mỗi bin tần số
        s = 0j
        for k in range(N):  # cộng theo thời gian
            s += x[k] * np.exp(-1j * 2*np.pi * m * k / N)
        X[m] = s

    return X

# demo: 2 sin
fs = 1000
N = 256
k = np.arange(N)
x = np.sin(2*np.pi*50*k/fs) + 0.5*np.sin(2*np.pi*120*k/fs)

print(x)
X = dft_1d(x)
mag = np.abs(X)

peak_bins = np.argsort(mag)[-10:][::-1]
print("Top bins:", peak_bins)
