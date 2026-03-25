import numpy as np

# mel bank
def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel2hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_bank(num_mel=12, n_fft=512, sr=16000):
    fmin, fmax = 0, sr/2
    mel_points = np.linspace(hz2mel(fmin), hz2mel(fmax), num_mel + 2)
    hz_points = mel2hz(mel_points)
    fftbin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    filter_bank = np.zeros((num_mel, n_fft // 2 + 1))
    for m in range(1, num_mel+1):
        fminus, fmid, fplus = fftbin_points[m-1], fftbin_points[m], fftbin_points[m+1]
        for k in range(fminus, fmid):
            filter_bank[m-1, k] = (k - fminus) / (fmid - fminus)
        for k in range(fmid, fplus):
            filter_bank[m-1, k] = (fplus - k) / (fplus - fmid)
    return filter_bank


# DCT coefficient
def dct_matrix(num_mel=12, mfcc_out=12):
    D = np.zeros((num_mel, mfcc_out))
    for m in range(num_mel):
        for n in range(mfcc_out):
            D[m, n] = np.cos(n * np.pi * (m + 0.5) / num_mel)
    return D

# dtw
def dist(x, y):
    dist = np.sum(np.abs(x - y)**2)
    return dist
def dtw_cost(sequence1, sequence2):
    w, h = sequence1.shape[0], sequence2.shape[0]
    D = np.zeros((w + 1, h + 1))
    D[0, 1:] = np.inf
    D[1:, 0] = np.inf
    for i in range(w):
        for j in range(h):
            D[i+1, j+1] = dist(sequence1[i, :], sequence2[j, :])
            D[i+1, j+1] += min(D[i, j+1], D[i+1, j], D[i, j])
    return D[w, h]
    