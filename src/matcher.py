import streamlit as st
import os
from utils import compare, generate, plotly
import timeit
import pandas as pd

st.set_page_config(layout="wide")
st.header("Impedance Matcher")

upload = st.checkbox("Upload files")
generate = st.checkbox("Generate synthetic data")

if generate:
    mu = st.slider("mean for noise", 0, 10, 0)
    sigma = st.slider("std dev for noise", 0.0, 1.0, 0.05)
    samples = st.slider("number of samples", 10, 1000, 100)
    
    if st.button("Generate"):
        st.write(mu)
        st.write(sigma)
        st.write(samples)
elif upload:
    base = st.file_uploader("Upload a baseline measurement", type=["csv"])
    sub = st.file_uploader("Upload a subsequent measurement", type=["csv"])
    assert base is not None
    assert sub is not None

else:
    dataDir = os.path.join(os.path.dirname(os.getcwd()), "data")
    files = [f for f in os.listdir(dataDir) if f[-4:] == ".csv"]
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
        

    