

import os
import glob
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.signal import find_peaks
from reedsolo import RSCodec

import streamlit as st


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

def hamming_distance(b1, b2):
    """
    Computes the Hamming distance between two bytearrays.
    """
    if len(b1) != len(b2):
        raise ValueError("Bytearrays must have same length")

    distance = 0
    for i in range(len(b1)):
        distance += bin(b1[i] ^ b2[i]).count('1')

    return distance, distance / (len(b1) * 8)

cwd = os.path.dirname(os.getcwd())
data_path = os.path.join(cwd, "data")
object_name = [x for x in os.listdir(data_path) if x != ".DS_Store"]

selected_object = st.sidebar.selectbox("Select Object Type for Similarity Analysis", object_name)

# if selected object has directories in it (i.e. it's a folder of folders)

sel_obj_path = os.path.join(data_path, selected_object)
object_listdir = os.listdir(sel_obj_path)
# print(object_listdir)
# print(os.path.isdir(object_listdir[0]))
if os.path.isdir(os.path.join(sel_obj_path, object_listdir[0])):  # if the first item in the list is a directory
    subdir_path = os.path.join(data_path, selected_object, "*/*.npy")
    npy_files = glob.glob(subdir_path)
else:
    npy_files = glob.glob(os.path.join(data_path, selected_object, "*.npy"))

print(npy_files)

npy_arrays = {f: np.load(f) for f in npy_files}

labels = [f.split("/")[-1].split(".")[0] for f in npy_files]
freq = [npy_arrays[f][:, 0] for f in npy_files]
responses = [npy_arrays[f][:, 1] for f in npy_files]


responses_w_complex = [npy_arrays[f][:, 1:3] for f in npy_files]

prominence_avg_diffs = []

rs = RSCodec(10)

# baseline is just the first signature taken
encoded_baseline = rs.encode(signal_to_byte_array(responses[0]))

fig = sp.make_subplots(rows=2, cols=1, subplot_titles=("Frequency Domain w peaks", "Avg Peak Prominence Diff by Signal"))

for idx, f_r in enumerate(zip(freq, responses)):
    f, r = f_r
    fig.add_trace(go.Scatter(x=f, y=r, name=labels[idx]), row=1, col=1)

    peak_idx_list, properties = find_peaks(r, prominence=0.1)

    #print("current idx: ", idx)
    # print("peak idxs: ", peak_idx_list)
    #print("peak properties: ", properties)

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

fig.add_trace(px.bar(x=labels, y=prominence_avg_diffs, labels={'x': 'Signal', 'y': 'Avg Peak Prominence Difference'}).data[0], row=2, col=1)
fig.update_layout(height=1000, width=1000)

st.plotly_chart(fig)

st.title("Subsample similarity")



num_subsamples = st.slider("Number of subsamples", 1, 50, 5)
similarity_threshold = st.slider("Threshold", 0.0, 0.5, 0.05)
list_results = st.checkbox("List Distance results")

response_pairs = [(responses[i], responses[j]) for i in range(len(responses)) for j in range(i + 1, len(responses))]
response_pair_labels = [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]

similarities = [similarity_metric(signal1, signal2, num_subsamples) for signal1, signal2 in response_pairs]

similar_pairs = [(response_pair_labels[i], similarities[i]) for i in range(len(similarities)) if similarities[i] < similarity_threshold]

if list_results:
    st.write(similar_pairs)

st.title("RS Decoding as String By Signal Name")

show_raw_encoding = st.checkbox("Show Pairwise Hamming Distance as Fraction of Total")

st.write(f"Standard baseline object is: {labels[0]}")

encoded_signals = [rs.encode(signal_to_byte_array(signal)) for signal in responses[1:]]  # skip the first with baseline
if show_raw_encoding:
    st.write([(l, hamming_distance(rs.decode(rsb), rs.decode(encoded_baseline))) for l, rsb in zip(labels, encoded_signals) if len(rsb) == len(encoded_baseline)])


st.title("Pairwise Cosine Similarity")

cosine_similarities = [np.dot(signal1, signal2) / (np.linalg.norm(signal1) * np.linalg.norm(signal2)) for signal1, signal2 in response_pairs if signal1 is not signal2 and len(signal1) == len(signal2)]
show_cosine_similarities = st.checkbox("Show Cosine Similarities")
if show_cosine_similarities:
    st.write([(response_pair_labels[i], cosine_similarities[i]) for i in range(len(cosine_similarities))])