

import os
import glob
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.signal import find_peaks
from reedsolo import RSCodec

# one level up from cwd
cwd = os.path.dirname(os.getcwd())
data_path = "data/LID/Bx5"

part_type = "LID"

# update for streamlit

lid_npy_files = glob.glob(os.path.join(cwd, data_path, "*.npy"))
lid_npy_arrays = {f: np.load(f) for f in lid_npy_files}

labels = [f.split("/")[-1].split(".")[0] for f in lid_npy_files]
freq = [lid_npy_arrays[f][:, 0] for f in lid_npy_files]
response = [lid_npy_arrays[f][:, 1] for f in lid_npy_files]

responses_w_complex = [lid_npy_arrays[f][:, 1:3] for f in lid_npy_files]

prominence_avg_diffs = []

rs = RSCodec(10)

def signal_to_byte_array(signal):
    # Normalize the signal to [0, 1], then scale it to [0, 255], and round the values to integers
    signal_normalized = (signal - signal.min()) / (signal.max() - signal.min())
    signal_int = np.round(signal_normalized * 255).astype(np.uint8)
    return signal_int.tobytes()

def subsample_signal(signal, num_samples):
    indices = np.linspace(0, len(signal) - 1, num_samples, dtype=int)
    return signal[indices]

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def similarity_metric(signal1, signal2, num_samples):
    subsampled_signal1 = subsample_signal(signal1, num_samples)
    subsampled_signal2 = subsample_signal(signal2, num_samples)
    pairwise_distances = [euclidean_distance(a, b) for a, b in zip(subsampled_signal1, subsampled_signal2)]
    aggregated_distance = np.mean(pairwise_distances)
    return aggregated_distance

encoded_signals = [rs.encode(signal_to_byte_array(signal)) for signal in response]
encoded_signals_2d = np.array([[encoded_signal[i] for i in range(2)] for encoded_signal in encoded_signals])







# make subplots 1 col, 3 rows
fig = sp.make_subplots(rows=3, cols=1, subplot_titles=("Frequency Domain w peaks", "RS codeword space", "Avg Peak Prominence Diff by Signal"))


for idx, f_r in enumerate(zip(freq, response)):
    f, r = f_r
    fig.add_trace(go.Scatter(x=f, y=r, name=labels[idx]), row=1, col=1)

    peak_idx_list, properties = find_peaks(r, prominence=0.1)

    print("current idx: ", idx)
    # print("peak idxs: ", peak_idx_list)
    # print("peak properties: ", properties)

    prominences = properties['prominences']
    avg_diff = np.mean(np.diff(prominences))
    prominence_avg_diffs.append(avg_diff)

    # ifft_t_signal = np.fft.ifft(responses_w_complex[idx])[:, 1].real
    # fig.add_trace(go.Scatter(x=f, y=ifft_t_signal), row=2, col=1)
    
    for pidx, peak_idx in enumerate(peak_idx_list):
        fig.add_annotation(
            x=f[peak_idx],
            y=r[peak_idx],
            hovertext=f"prominence: {properties['prominences'][pidx]}",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40,
        )

        fig.add_trace(px.scatter(x=encoded_signals_2d[:, 0], y=encoded_signals_2d[:, 1], labels={'x': 'd1', 'y': 'd2'}).data[0], row=2, col=1)

for i, (x, y) in enumerate(encoded_signals_2d):
    fig.add_annotation(
        x=x, y=y, text=f'Signal {i + 1}', showarrow=False, font=dict(size=12)
    )
    

fig.add_trace(px.bar(x=labels, y=prominence_avg_diffs, labels={'x': 'Signal', 'y': 'Avg Peak Prominence Difference'}).data[0], row=3, col=1)


fig.show()