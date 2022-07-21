# reference : https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python

import numpy as np

def noisy(data, snr):
    target_snr_db = snr
    sig_avg_pres = np.mean(data)
    sig_avg_db = 10 * np.log10(sig_avg_pres)
    
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_pres = 10 ** (noise_avg_db / 10)
    
    mean_noise = 0
    noise_pres = np.random.normal(mean_noise, np.sqrt(noise_avg_pres), size=(int(data.shape[0]), int(data.shape[1])))
    
    noise_data = data + noise_pres
    
    return noise_data