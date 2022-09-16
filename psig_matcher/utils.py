from audioop import mul
from curses import raw
import os
from re import I
from statistics import variance
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import simpson
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from perlin_noise import PerlinNoise
from itertools import combinations
import matplotlib.pyplot as plt
import glob

from psig_matcher import DATA_DIR, HEADER_LEN, ALL_PART_TYPES

def load_np(filename):
    with open(filename, 'rb') as f:
        return np.load(f)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


class PiezoelectricSignature:
    """ Represent a single measurement of a part type and instance over a range of frequencies
    """
    def __init__(self, part_type, instance_id, from_file=True, filename=None, raw_data=None):
        self.from_file = from_file
        if from_file:
            self.filename = filename
            self.load_data()
        else:
            #assert raw_data is not None
            self.data = raw_data
            self.load_data()
        #print(f"piezoelectric signature filename: {filename}")
        self.part_type = part_type
        self.instance_id = instance_id
        self.get_normalized_stats()

    def load_data(self):
        if self.from_file:
            data = load_np(self.filename)
        else:
            data = self.data
        self.freq = data[:, 0]
        self.min_freq = self.freq[0]
        self.max_freq = self.freq[-1]
        self.real_imp = data[:, 1]
        self.imag_imp = data[:, 2]
        self.size = len(data)
        self.mean = np.average(self.real_imp)
        self.std = np.std(self.real_imp)

    def normalize_signature(self):
        self.normed_freq = normalize(self.freq)
        self.normed_real_imp = normalize(self.real_imp)
        self.normed_imag_imp = normalize(self.imag_imp)

    def get_normalized_stats(self):
        self.normalize_signature()
        self.norm_avg = np.average(self.normed_real_imp)
        self.norm_std = np.std(self.normed_real_imp)
        self.norm_avg_imag = np.average(self.normed_imag_imp)
        self.norm_std_imag = np.std(self.normed_imag_imp)

    def generate_synthetic(self, n=50, noise_type='perlin', octaves=2, plot=True):
        """ Generate n noised signatures using Perlin noise. Return the list of mean and std for
            each generated signature """
        generated_real_imp_vals = []
        generated_imag_imp_vals = []
        noise = PerlinNoise(octaves=octaves, seed=42)
        for i in range(n):
            noise_val = np.array([10 * np.random.random_sample() * noise([i/(len(self.freq))]) for i in range(len(self.freq))])
            generated_real_imp_vals.append(self.real_imp + noise_val)
            generated_imag_imp_vals.append(self.imag_imp + noise_val)

        mus_real = [np.average(normalize(gen)) for gen in generated_real_imp_vals]
        sigmas_real = [np.std(normalize(gen)) for gen in generated_real_imp_vals]

        mus_imag = [np.average(normalize(gen)) for gen in generated_imag_imp_vals]
        sigmas_imag = [np.std(normalize(gen)) for gen in generated_imag_imp_vals]

        if plot:
            self.plot(list(zip(generated_real_imp_vals, generated_imag_imp_vals)), np.average(mus_real), np.average(sigmas_real), np.average(mus_imag), np.average(sigmas_imag))

        return np.average(mus_real), np.average(sigmas_real), np.average(mus_imag), np.average(sigmas_imag)

    def plot(self, signatures, mu_real, std_real, mu_imag, sigma_imag):
        """ used for visualizing generated signatures """
        plot_title = f"Synthetic signatures of {self.instance_id} (n={len(signatures)}), avg mean real/imag ={round(mu_real, 3)} ({round(self.norm_avg, 3)}) / {round(mu_imag, 3)} ({round(self.norm_avg_imag)}), avg std real/imag ={round(std_real, 3)} ({round(self.norm_std, 3)}) / {round(sigma_imag, 3)} ({round(self.norm_std_imag)})"

        plt.plot(self.freq, self.real_imp, label=f"{self.part_type}_{self.instance_id}_real")
        plt.plot(self.freq, self.imag_imp, label=f"{self.part_type}_{self.instance_id}_imag")

        for idx, sig in enumerate(signatures):
            #plot_label = f"generated_{idx}"
            plt.plot(self.freq, sig[0])
            plt.plot(self.freq, sig[1])

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Real Impedance (Ohms) - log scale")
        plt.title(plot_title)
        plt.legend()
        #plt.yscale('log')
        plt.show()

    def __repr__(self):
        q = f" ++ Summary of {self.part_type} instance {self.instance_id} ++\n"
        f = f"file: {self.filename}\n"
        mi = f"min freq: {self.min_freq}\n"
        ma = f"max freq: {self.max_freq}\n"
        s = f"size: {self.size}\n"
        nm = f"normalized mean: {self.norm_avg}\n"
        ns = f"normalized std: {self.norm_std}\n"
        e = "++++++++++++++++++++++++++++++++++\n"
        return q + f + mi + ma + s + nm + ns + e

class PartInstance:
    """ Represent a single instance of a part type.
        A single instance can have multiple signatures
    """
    def __init__(self, data_loc, part_type, instance_id):
        self.part_type = part_type
        self.instance_id = instance_id
        try:
            self.data_files = glob.glob(f"{os.path.join(data_loc, instance_id)}/*.npy")
            #print(self.data_files)
        except:
            print(f"Error invalid instance id {instance_id}")

        self.load_signatures()
        self.get_average_signature()

    def load_signatures(self):
        signatures = {}

        for data_file in self.data_files:
            signature_name = data_file.split("/")[-1]
            ps = PiezoelectricSignature(self.part_type, self.instance_id, filename=data_file)
            signatures[signature_name[:-4]] = ps #{"freq": ps.freq, "real": ps.real_imp, "imag": ps.imag_imp}

        self.signatures = signatures
    
    def list_signatures(self):
        print(f"signature list for {self.part_type} instance {self.instance_id}: {self.signatures.keys()}")

    def get_signature(self, sig_id):

        sig_name = f"{self.part_type.lower()}_{self.instance_id}_{sig_id}"
        try:
            return self.signatures[sig_name]
        except KeyError as e:
            print(f"invalid signature name {sig_name}")
        

    def get_average_signature(self, min_freq=10000, max_freq=150000):
        """ obtain the average real response values from all signatures 
            of a single instance that fall within min_freq and max_freq """

        real_response_vals = []
        imag_response_vals = []
        freq_range = None
        avg_signature = None

        for sig in self.signatures.values():
            freq = sig.freq # ['freq']
            real = sig.real_imp # ['real']
            imag = sig.imag_imp
            # imag = sig['imag']
            if min_freq in freq and max_freq in freq:
                # only consider signatures that fall into the specified range
                if freq_range is None:
                    freq_range = freq
                real_response_vals.append(real)
                imag_response_vals.append(imag)
            else:
                continue
                
        
        if real_response_vals and imag_response_vals:
            real_response_np = np.vstack(np.array(real_response_vals)).transpose()
            imag_response_np = np.vstack(np.array(imag_response_vals)).transpose()

            avg_real_response_vals = np.average(real_response_np, axis=-1)
            avg_imag_response_vals = np.average(imag_response_np, axis=-1)

            avg_signature = np.column_stack((freq_range, avg_real_response_vals, avg_imag_response_vals))
        
        if avg_signature is not None:
            self.avg_signature = PiezoelectricSignature(self.part_type, self.instance_id, from_file=False, raw_data=avg_signature)

    def plot(self, include_avg=True, plot_title=None):

        plot_title = f"All signatures of {self.instance_id}"

        for filename, signature in self.signatures.items():
            plot_label = filename[:-4]
            plt.plot(signature.freq, signature.real_imp, label=plot_label + "_real")
            plt.yscale('log')

        if include_avg and self.avg_signature is not None:
            plt.plot(self.avg_signature[:, 0], self.avg_signature[:, 1], label=plot_label + "_avg")

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Real Impedance (Ohms) - log scale")
        plt.title(plot_title)
        plt.legend()
        plt.show()

class Part:
    """ Represent many instances of a given part type, e.g. Box
        A single part type can have multiple instances
    """
    def __init__(self, part_type):
        assert part_type in ALL_PART_TYPES, f"part type {part_type} is not valid"
        self.part_type = part_type
        self.data_loc = os.path.join(DATA_DIR, part_type)
        self.instance_names = sorted([inst_name for inst_name in os.listdir(self.data_loc) if inst_name != ".DS_Store"])

        self.instances = {i: PartInstance(self.data_loc, self.part_type, i) for i in self.instance_names}
        self.instance_count = len(self.instances)
        
    def list_instances(self):
        print(f"instance names for part type {self.part_type}: {self.instance_names}")
        # for i in self.instances:
        #     print(f"number of signatures for part type {self.part_type} instance {i}: {len(self.instances[i].signatures)}")

    def plot(self):
        for i in self.instances.values():
            i.plot()

    def get_instance(self, instance_id):
        try:
            return self.instances[instance_id]
        except:
            print(f"Error invalid instance id: {instance_id}")


class Comparator:
    """ Class to compare two or more signatures - assumes the frequency range for measurements
        is the same (i.e. 10-100kHz for both signatures) or can be interpolated accordingly
    """
    def __init__(self, sig1, sig2):
        self.sig1 = sig1
        self.sig2 = sig2
        pass

    def compare(self):
        print(f"comparing {self.sig1.part_type}_{self.sig1.instance_id} with {self.sig2.part_type}_{self.sig2.instance_id}")
        mse = mean_squared_error(self.sig1.real_imp, self.sig2.real_imp)
        rmse = np.sqrt(mse)  # same as RMSD
        l1 = np.average(self.sig1.real_imp - self.sig2.real_imp)  # avg pointwise distance between response vals
        metrics = {'mse': mse, 'rmse': rmse, 'l1': l1}
        print(f"comparison metrics: {metrics}")

def compare(instance_1, instance_2, metrics=["l1", "l2", "hamming", "rmsd"]):
    """ compare two psigs with each of metric, return a dictionary of their distances """
    pass

def generate_with_noise(ps, n_sigs=50, noise_type=["perlin"], octaves=8, seed=42, real_only=False):
    """ generate n_sigs given a real piezoelectric signature with noise_type noise"""
    pass

