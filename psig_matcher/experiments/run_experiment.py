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
import mlflow
from sklearn.model_selection import ParameterGrid
import multiprocessing as mp
import time
import random
import warnings
import traceback

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback
warnings.simplefilter("always")

@dataclass
class MultivariateNormalDistribution:
    mean: np.ndarray
    cov: np.ndarray

@dataclass
class Part:

    type: str
    sub_part_name: str
    sensor: str
    signals: List   # Signal is numpy array of (500,3) with [frequency, Z, X]


def load_part_data(part_type: str) -> List[Part]:

    parts = []
    for part_dir in os.listdir(f'psig_matcher/data/{part_type}'):
        if not os.path.isdir(f'psig_matcher/data/{part_type}/{part_dir}'): continue
        sensor = part_dir[1:]
        measurement_files = glob.glob(f'psig_matcher/data/{part_type}/{part_dir}/*.npy')
        if not measurement_files: continue
        measurements = [np.load(f) for f in measurement_files]
        parts.append(Part(part_type, part_dir, sensor, measurements))

    return parts

def limit_deminsionality(parts: List[Part], frequeny_indexes: List[int]) -> List[Part]:
    """Use only a subset of the frequencies for the analysis. This effectivley transforms the
    500 dimension multivariant distribution to a n-dimentional distribution where n is the
    length of the frequency_indexes.

    Further, this assumes use of the X axis"""

    return [
        dataclasses.replace(part, signals=[[signal[index][1] for index in frequeny_indexes] for signal in part.signals])
        for part in parts]

def compute_normal_ci(x: List[float], confidence: float) -> Tuple[float, float]:
    """Computes the confidence interval for a given confidence bound."""

    if np.all(np.isclose(x, 0)):

        if np.array(x).ndim==1:
            return 0,0
        elif np.array(x).ndim==2:
            return np.zeros(np.array(x).shape[1]), np.zeros(np.array(x).shape[1])
        else:
            raise ValueError('x must be 1 or 2 dimensional')

    if len(x) < 30:
        return st.t.interval(confidence, len(x)-1, loc=np.mean(x, axis=0), scale=np.std(x, axis=0))
    else:
        return stats.norm.interval(confidence, loc=np.mean(x, axis=0), scale=np.std(x, axis=0))

def estimate_normal_dist(x: List[float], confidence: float) -> MultivariateNormalDistribution:
    """Estimate the normal distribution for the given data.
    This is done using: https://handbook-5-1.cochrane.org/chapter_7/7_7_3_2_obtaining_standard_deviations_from_standard_errors_and.htm#:~:text=The%20standard%20deviation%20for%20each,should%20be%20replaced%20by%205.15.
    """

    data_mean = np.mean(x, axis=0)
    data_std = np.std(x, axis=0)

    t_score = st.t.ppf(confidence, np.array(x).shape[0] - 1)
    z_score = st.norm.ppf(confidence)
    scaling_factor = t_score if len(x) < 30 else z_score

    desired_std = data_std + (data_std * scaling_factor)
    derived_data = (x - data_mean) / data_std * desired_std + data_mean
    derived_data_cov = np.cov(derived_data, rowvar=False)

    regularization_term = 1e-3
    if derived_data_cov.shape[0] >= derived_data.shape[0]:
        derived_data_cov += np.eye(derived_data_cov.shape[0]) * regularization_term

    return MultivariateNormalDistribution(data_mean, derived_data_cov)

def probability_of_multivariant_point(mu: np.ndarray, cov: np.ndarray, x: np.ndarray) -> float:

    #https://stats.stackexchange.com/questions/331283/how-to-calculate-the-probability-of-a-data-point-belonging-to-a-multivariate-nor
    # Double check this math
    m_dist_x = np.dot((x-mu).transpose(),np.linalg.inv(cov))
    m_dist_x = np.dot(m_dist_x, (x-mu))
    return 1-stats.chi2.cdf(m_dist_x, 3)

def estimate_overlap_of_set_with_sample_signals(parts: List[Part], samples: int, meta_pdf_ci: float, part_pdf_ci: float, confidence_bound: float) -> float:
    """ I believe this is the best solution out of all them. We are directly modeling the distribution/state space that
    the signals come from, and sampling from that. This directly correlates with the CI and is intuitive. See notion for
    more details and defense. """

    min_confidence = 1 - confidence_bound
    signals = [
        signal for part in parts
        for signal in part.signals]

    part_pdfs = [estimate_normal_dist(part.signals, part_pdf_ci) for part in parts]
    sample_pdf = estimate_normal_dist(signals, meta_pdf_ci)
    state_space_samples = np.random.multivariate_normal(sample_pdf.mean, sample_pdf.cov, samples)

    sample_confidences = [
        [probability_of_multivariant_point(pdf.mean, pdf.cov, sample) for pdf in part_pdfs]
        for sample in state_space_samples]

    filtered_confidences = [
        list(filter(lambda confidence: confidence >= min_confidence, sample_confidence))
        for sample_confidence in sample_confidences]

    # We're ok with up to 1 match, but every one more than that is a conflict.
    collisions = [max(len(confidences)-1, 0) for confidences in filtered_confidences]
    return sum(collisions)/(samples*len(part_pdfs))

def run_meta_markov_multivariant_analysis(client: mlflow.tracking.MlflowClient, run_id: int, parts: List[Part], part_dim: int, num_samples: int, meta_pdf_ci: float, part_pdf_ci: float, confidence_bound: float):
    """ Runs the Monte Carlo Approximation of multivariant collision using the signal sample meta
    pdf methodology. The Monte Carlo Approximation will continually be run until the confidence interval
    converges and the average of the previous 10 runs is not smaller than the average of the previous 100 runs."""

    collisions = []
    confidence_ranges = []
    while True:

        multivariant_parts = limit_deminsionality(parts, list(range(part_dim)))
        collision_rate = estimate_overlap_of_set_with_sample_signals(multivariant_parts, num_samples, meta_pdf_ci, part_pdf_ci, confidence_bound)

        collisions.append(collision_rate)
        # client.log_metric(run_id, "monte_carlo_collision_rate", collision_rate)

        if len(collisions) < 2: continue
        lower, upper = compute_normal_ci(collisions, 0.95)

        confidence_ranges.append(upper - lower)
        # client.log_metric(run_id, "monte_carlo_confidence_interval", upper - lower)
        # print(len(confidence_ranges))

        if len(confidence_ranges) > 100 and np.mean(confidence_ranges[-10:]) >= np.mean(confidence_ranges[-100:]):
            return upper

def simulate_part_pdf_convergence(part_signals: np.ndarray, part_dim: int, meta_pdf_ci: float, part_pdf_ci: float):
    """ Given a discrete set of signals, this will simulate the part PDF CI convergence methodology.
    This function pretend that the set of the signals is infinite, Also incorporate logic to handle
    if we didn't converge before we ran out of data. Have modular connection style such that we could
    add streaming data source in the future. """
    sub_samples = []
    confidence_ranges = []
    while len(part_signals) != 0:
        # The below segment is equivalent to sub_samples.append(part_signals.pop())
        part_sig = part_signals[0:5]
        part_signals = part_signals[6:]
        sub_samples += part_sig
        
        lower, upper = compute_normal_ci(sub_samples, part_pdf_ci)
        confidence_ranges.append(upper - lower)
        
        mlflow.log_metric("part_pdf_confidence_interval", np.mean(upper - lower))

        if len(confidence_ranges) > 100 and np.mean(confidence_ranges[-10:]) >= np.mean(confidence_ranges[-100:]):
            return upper

def run_experiment(experiment_id: int, part_type: str, part_dim: int, meta_pdf_ci: float, part_pdf_ci: float):

    print("in run experiment")
    client = mlflow.tracking.MlflowClient()
    run_id = client.create_run(experiment_id).info.run_id

    client.log_param(run_id, "part_type", part_type)
    client.log_param(run_id, "part_dim", part_dim)
    client.log_param(run_id, "meta_pdf_ci", meta_pdf_ci)
    client.log_param(run_id, "part_pdf_ci", part_pdf_ci)
    
    con_parts = load_part_data(part_type)
    con_parts = limit_deminsionality(con_parts, list(range(part_dim)))
    for con_part in con_parts:
        print(con_part.sub_part_name)
        upper_collision_rate = simulate_part_pdf_convergence(con_part.signals, part_dim, part_pdf_ci)
        client.log_metric(run_id, "part_pdf_convergence_upper_collision_rate", upper_collision_rate)
        print("final upper collision rate -", upper_collision_rate)


# def main():
#     """ This script can be run as such:
#     python3 psig_matcher/experiments/run_experiment.py --part_type=CONTAINER --part_dim=5 --num_samples=100 --meta_pdf_ci=0.999 --part_pdf_ci=0.999 --confidence_bound=0.999 """

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--part_type', type=str, required=True)
#     parser.add_argument('--part_dim', type=int, required=True)
#     parser.add_argument('--num_samples', type=int, required=True)
#     parser.add_argument('--meta_pdf_ci', type=float, required=True)
#     parser.add_argument('--part_pdf_ci', type=float, required=True)
#     parser.add_argument('--confidence_bound', type=float, required=True)

#     args = parser.parse_args()
#     run_experiment(args.part_type, args.part_dim, args.num_samples, args.meta_pdf_ci, args.part_pdf_ci, args.confidence_bound)

def run_parallel_experiment():

    mlflow.set_experiment("Experiment 1")
    experiment_id = mlflow.get_experiment_by_name(name='Experiment 1').experiment_id

    # part_types = ["BEAM", "BOX", "BRK", "CON", "CONLID", "FLG", "IMP", "LID", "SEN", "TUBE", "VNT"]

    part_types=["BEAM", "CONTAINER", "CONLID", "LID", "SEN", "TUBE"]
    part_dim=2
    meta_pdf_ci=0.95
    part_pdf_ci=0.95   
    for part_type in part_types:   
        run_experiment(experiment_id, part_type, part_dim, meta_pdf_ci, part_pdf_ci)


if __name__ == '__main__':
    run_parallel_experiment()
