import streamlit as st
import os
from utils import compare, generate, plotly
import timeit
import pandas as pd

st.set_page_config(layout="wide")
st.header("Impedance Matcher")


def listFiles():
    dataDir = os.path.join(os.path.dirname(os.getcwd()), "data")
    return [f for f in os.listdir(dataDir) if f[-4:] == ".csv"]
files = listFiles() 

if st.checkbox("Generate synthetic data"):
    sigma = st.number_input('enter a sigma for stdev of noise', min_value=0.0, max_value=0.5, step=0.01)
    st.write(f"sigma: {sigma}")
    # smaller sigma --> same instance different measurement
    # higher sigma --> different instance
    samples = st.slider("number of samples", 2, 500, 10)
    file = st.selectbox("Select a measurement to add noise to", files)
    
    if st.button("Generate"):
        st.write(sigma)
        st.write(samples)
        # todo how to smooth noise
        noise, plt = generate(file, samples, sigma)
        st.plotly_chart(plt)

    # todo add metrics and histogram of pairwise HD 
    if st.button("Pairwise metrics"):
        st.write("pairwise metrics")

elif st.checkbox("Upload files"):
    base = st.file_uploader("Upload a baseline measurement", type=["csv"])
    sub = st.file_uploader("Upload a subsequent measurement", type=["csv"])
    assert base is not None
    assert sub is not None
else:
    base = st.selectbox("Select a baseline measurement", files)
    sub = st.selectbox("Select a subsequent measurement", files)

    if base and sub and st.button("Compare"):
        with st.spinner("comparing..."):
            s = timeit.timeit()
            pl = plotly(base, sub)
            cmp, base, sub = compare(base, sub)
            e = timeit.timeit()
        st.write(f"done in {s-e}s")

        c1, c2, c3 = st.beta_columns((1, 1, 2))
        c3.plotly_chart(pl)

        c1.write(f"peak base raw : {cmp.base.peakFrq}") # frequencies
        c1.write(f"peak base prom : {cmp.base.prom}") # prominences
        c1.write(f"peak sub raw : {cmp.sub.peakFrq}") # frequencies
        c1.write(f"peak sub prom : {cmp.sub.prom}") # prominences
        c1.write(f"peak base data: {cmp.peakBase}")
        c1.write(f"peak sub data: {cmp.peakSub}")
        c1.write(f"peak target key: {cmp.peakTargetKey}")
        c1.write(f"peak actual key: {cmp.peakActualKey}")

        if cmp.peakTargetKey == cmp.peakActualKey:
            c1.success("Peak Match")
        else:
            c1.error("Peak mismatch")

        c2.write(f"imp base data: {cmp.impBase}")
        c2.write(f"imp sub data: {cmp.impSub}")
        c2.write(f"imp target key: {cmp.impTargetKey}")
        c2.write(f"imp actual key: {cmp.impActualKey}")
            
        if cmp.impTargetKey == cmp.impActualKey:
            c2.success("Impedance Match")
        else:
            c2.error("Impedance mismatch")

        for k, v in cmp.metrics.items():
            st.write(k, v) 
        

    