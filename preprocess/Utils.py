import mne
import numpy as np
from scipy import signal

def read_edf_file(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_channels(['PSG_F3', 'PSG_F4', 'PSG_C3', 'PSG_C4', 'PSG_O1', 'PSG_O2'])

    return raw

def filter_band(raw, l_freq=1, h_freq=50, notch_freq=50):
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    
    raw.notch_filter(freqs=notch_freq)
    
    return raw

def remove_artifacts_ica(raw, n_components=15):
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)

    ica.fit(raw)
    
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    
    raw_cleaned = raw.copy()
    ica.apply(raw_cleaned)
    
    return raw_cleaned