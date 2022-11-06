from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import os
import glob
import scipy.stats as st
from scipy import stats
import dataclasses
import argparse
import sys
from mlflow import log_metric, log_param, log_artifacts

@dataclass
class NormalDistribution
    mean float
    std float
    
@dataclass
class Part
    
    type str
    sub_part_name str
    sensor str
    signals List   # Signal is numpy array of (500,3) with [frequency, Z, X]
    
    
def load_part_data(part_type str) - List[Part]
    
    parts = []
    for part_dir in os.listdir(f'psig_matcherdata{part_type}')
        
        sensor = part_dir[1]
        measurement_files = glob.glob(f'psig_matcherdata{part_type}{part_dir}.npy')
        measurements = [np.load(f) for f in measurement_files]
        parts.append(Part(part_type, part_dir, sensor, measurements))
    
    return parts

def limit_deminsionality(parts List[Part], frequeny_indexes List[int]) - List[Part]
    Use only a subset of the frequencies for the analysis. This effectivley transforms the 
    500 dimension multivariant distribution to a n-dimentional distribution where n is the
    length of the frequency_indexes.
    
    Further, this assumes use of the X axis
    
    return [
        dataclasses.replace(part, signals=[[signal[index][1] for index in frequeny_indexes] for signal in part.signals])
        for part in parts]

def compute_normal_ci(x List[float], confidence float) - Tuple[float, float]
    Computes the confidence interval for a given confidence bound.
    
    if sum(x) == 0 return (0, 0)
    
    if len(x)  30
        return st.t.interval(confidence, len(x)-1, loc=np.mean(x), scale=st.sem(x))
    else
        return stats.norm.interval(confidence, loc=np.mean(x), scale=np.std(x))

def estimate_normal_dist(x List[float], confidence float) - NormalDistribution
    Estimate the normal distribution for the given data.
    This is done using httpshandbook-5-1.cochrane.orgchapter_77_7_3_2_obtaining_standard_deviations_from_standard_errors_and.htm#~text=The%20standard%20deviation%20for%20each,should%20be%20replaced%20by%205.15.
    
    
    
    val_comp = st.t.ppf if len(x)  30 else stats.norm.ppf
    lower, upper = compute_normal_ci(x, confidence)
   
    val = val_comp(confidence, len(x)-1)
    std = np.sqrt(len(x))(upper-lower)val
    return NormalDistribution(np.mean(x, axis=0), std)
   

def probability_of_multivariant_point(mu List[float], cov List[List[float]], x List[float]) - float
    
    #httpsstats.stackexchange.comquestions331283how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor
    # Double check this math
    m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(cov))
    m_dist_x = np.dot(m_dist_x, (x-mu))
    return 1-stats.chi2.cdf(m_dist_x, 3)

def estimate_overlap_of_set_with_sample_signals(parts List[Part], samples int, meta_pdf_ci float, part_pdf_ci float, confidence_bound float) - float
     I believe this is the best solution out of all them. We are directly modeling the distributionstate space that
    the signals come from, and sampling from that. This directly correlates with the CI and is intuitive. See notion for
    more details and defense. 
    
    min_confidence = 1 - confidence_bound
    signals = [
        signal for part in parts 
        for signal in part.signals]
    
    part_pdfs = [estimate_normal_dist(part.signals, part_pdf_ci) for part in parts]
    sample_pdf = estimate_normal_dist(signals, meta_pdf_ci)
    
    state_space_samples = np.random.multivariate_normal(sample_pdf.mean, np.diag(sample_pdf.std), samples)
    
    # using probability_of_multivariant_point no longer directly equates to false negative rate.
    # TODO (henry) Figure out relationship between integrated pdf range and false negative rate
    sample_confidences = [
        [probability_of_multivariant_point(pdf.mean, np.diag(pdf.std), sample) for pdf in part_pdfs]
        for sample in state_space_samples]

    filtered_confidences = [
        list(filter(lambda confidence confidence = min_confidence, sample_confidence))
        for sample_confidence in sample_confidences]

    # We're ok with up to 1 match, but every one more than that is a conflict.
    collisions = [max(len(confidences)-1, 0) for confidences in filtered_confidences]
    return sum(collisions)(sampleslen(part_pdfs))

def run_meta_markov_multivariant_analysis(parts List[Part], part_dim int, num_samples int, meta_pdf_ci float, part_pdf_ci float, confidence_bound float)
     Runs the Monte Carlo Approximation of multivariant collision using the signal sample meta
    pdf methodoly. The Monte Carlo Approximation will continually be run until the confidence interval
    converges and the average of the previous 10 runs is not smaller than the average of the previous 100 runs.
    
    collisions = []
    confidence_ranges = []
    while True
        
        multivariant_parts = limit_deminsionality(parts, list(range(part_dim)))
        collision_rate = estimate_overlap_of_set_with_sample_signals(multivariant_parts, num_samples, meta_pdf_ci, part_pdf_ci, confidence_bound)
        log_metric("collision_rate", collision_rate)
            
        collisions.append(collision_rate)
        lower, upper = compute_normal_ci(collisions, 0.95)
        confidence_ranges.append(upper - lower)
        log_metric("ci", upper)
        # print(fEstimated collision rate from sample distributiion has range {upper - lower})
        
        if len(confidence_ranges)  100 and np.mean(confidence_ranges[-10]) = np.mean(confidence_ranges[-100])
            # final approximated collision rate, and its associated CI
            log_metric("final_collision_rate", collision_rate)
            log_metric("final_ci", upper)
            return upper
    
def run_experiment(part_type str, part_dim int, num_samples int, meta_pdf_ci float, part_pdf_ci float, confidence_bound float)

    con_parts = load_part_data(part_type)
    upper_collision_rate = run_meta_markov_multivariant_analysis(con_parts, part_dim, num_samples, meta_pdf_ci, part_pdf_ci, confidence_bound)
    print(fUpper collision rate {upper_collision_rate  100}%)

def main()
     This script can be run as such
    python3 psig_matcherexperimentsrun_experiment.py --part_type=CON --part_dim=5 --num_samples=100 --meta_pdf_ci=0.999 --part_pdf_ci=0.999 --confidence_bound=0.999 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_type', type=str, required=True)
    parser.add_argument('--part_dim', type=int, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    parser.add_argument('--meta_pdf_ci', type=float, required=True)
    parser.add_argument('--part_pdf_ci', type=float, required=True)
    parser.add_argument('--confidence_bound', type=float, required=True)
    
    args = parser.parse_args()
    # capture the system hyperparameters that were used when running the experiment 
    log_param("part_type", args.part_type)
    log_param("part_dim", args.part_dim)
    log_param("num_samples", args.num_samples)
    log_param("meta_pdf_ci", args.meta_pdf_ci)
    log_param("part_pdf_ci", args.part_pdf_ci)
    log_param("confidence_bound", args.confidence_bound)
    run_experiment(args.part_type, args.part_dim, args.num_samples, args.meta_pdf_ci, args.part_pdf_ci, args.confidence_bound)
    
if __name__ == '__main__'
    main()
