def estimate_overlap_of_set_with_meta_pdf(pdfs: List[NormalDistribution], samples: int, confidence_bound: float) -> float:
    """Estimates the overlap between a set of distributions.
    
    The meta pdf is really just the combined pdfs of all the distributions, then we're drawing from that
    and seeing how many samples would cause conflicts. How can we prove the distribution we're pulling samples
    from is representative of the entire population? Is the estimated confidence good enough? 
    
    Could we potentially randomly sample from each distribution and just see which ones end up overlapping
    with the other distributions? TODO (henry): Think about this more 
    
    TODO (henry): This takes so long, like 5's per sample. We need to run 10,000's samples many times. 
    Need to computationall optimize this.
    """
    
    pdf_means = [pdf.mean for pdf in pdfs]    
    min_confidence = 1 - confidence_bound
    meta_pdf = estimate_normal_dist(pdf_means, 0.95)
    meta_samples = np.random.multivariate_normal(meta_pdf.mean, np.diag(meta_pdf.std), samples)
    
    sample_confidences = [
        [mvn(mean=pdf.mean, cov=pdf.std).cdf(sample) 
        for pdf in pdfs]
        for sample in meta_samples]
    
    filtered_confidences = [
        list(filter(lambda confidence: confidence >= min_confidence, sample_confidence))
        for sample_confidence in sample_confidences]

    # We're ok with up to 1 match, but every one more than that is a conflict.
    collisions = [1 for confidences in filtered_confidences if len(confidences) > 1]
    return sum(collisions)/samples
    
    
    
