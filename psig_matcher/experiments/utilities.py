from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as st
from statistics import NormalDist
from scipy.stats import multivariate_normal as mvn
import time
from scipy import stats
import multiprocessing as mp
import dataclasses


@dataclass
class NormalDistribution:
    mean: float
    std: float
    
@dataclass
class Part:
    
    type: str
    sub_part_name: str
    sensor: str
    signals: List   # Signal is numpy array of (500,3) with [frequency, Z, X]

def limit_deminsionality(parts: List[Part], frequeny_indexes: List[int]) -> List[Part]:
    """Use only a subset of the frequencies for the analysis. This effectivley transforms the 
    500 dimension multivariant distribution to a n-dimentional distribution where n is the
    length of the frequency_indexes.
    
    Further, this assumes use of the X axis"""
    
    return [
        dataclasses.replace(part, signals=[[signal[index][1] for index in frequeny_indexes] for signal in part.signals])
        for part in parts]

def estimate_normal_dist(x: List[float], confidence: float) -> NormalDistribution:
    """Estimate the normal distribution for the given data.
    This is done using: https://handbook-5-1.cochrane.org/chapter_7/7_7_3_2_obtaining_standard_deviations_from_standard_errors_and.htm#:~:text=The%20standard%20deviation%20for%20each,should%20be%20replaced%20by%205.15.
    
    TODO (henry): I'm not sure this is correct.
    """
    
    # Use T distribution for small sample sizes
    if len(x) < 30:
        lower, upper = st.t.interval(confidence, len(x)-1, loc=np.mean(x), scale=st.sem(x))
        t_value = st.t.ppf(confidence, len(x)-1)
        std = np.sqrt(len(x))*(upper-lower)*t_value
    
    # Use normal distribution for larger sample sizes
    else:
        lower, upper = st.norm.interval(confidence, loc=np.mean(x), scale=st.sem(x))
        z_value = st.norm.ppf(confidence)
        std = np.sqrt(len(x))*(upper-lower)*z_value
    
    return NormalDistribution(np.mean(x, axis=0), std)

def probability_of_multivariant_point(mu: List[float], cov: List[List[float]], x: List[float]) -> float:
    
    #https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor
    # Double check this math
    m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(cov))
    m_dist_x = np.dot(m_dist_x, (x-mu))
    return 1-stats.chi2.cdf(m_dist_x, 3)

def estimate_overlap_of_set_with_sample_signals(parts: List[Part], samples: int, confidence_bound: float) -> float:
    """ I believe this is the best solution out of all them. We are directly modeling the distribution/state space that
    the signals come from, and sampling from that. This directly correlates with the CI and is intuitive. See notion for
    more details and defense. """
    
    min_confidence = 1 - confidence_bound
    signals = [
        signal for part in parts 
        for signal in part.signals]
    
    part_pdfs = [estimate_normal_dist(part.signals, 0.95) for part in parts]
    #part_mvn_pdfs = [mvn(mean=pdf.mean, cov=pdf.std) for pdf in part_pdfs]
    sample_pdf = estimate_normal_dist(signals, 0.95)
    
    state_space_samples = np.random.multivariate_normal(sample_pdf.mean, np.diag(sample_pdf.std), samples)
    sample_confidences = [
        [probability_of_multivariant_point(pdf.mean, np.diag(pdf.std), sample) for pdf in part_pdfs]
        for sample in state_space_samples]

    filtered_confidences = [
        list(filter(lambda confidence: confidence >= min_confidence, sample_confidence))
        for sample_confidence in sample_confidences]

    # We're ok with up to 1 match, but every one more than that is a conflict.
    collisions = [max(len(confidences)-1, 0) for confidences in filtered_confidences]
    return sum(collisions)/(samples*len(part_pdfs))

def load_part_data(part_type: str) -> List[Part]:
    
    parts = []
    for part_dir in os.listdir(f'psig_matcher/data/{part_type}'):
        
        sensor = part_dir[1:]
        measurement_files = glob.glob(f'psig_matcher/data/{part_type}/{part_dir}/*.npy')
        measurements = [np.load(f) for f in measurement_files]
        parts.append(Part(part_type, part_dir, sensor, measurements))
    
    return parts


