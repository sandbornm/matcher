import streamlit as st
import os
from utils import compare
import timeit

st.set_page_config(layout="wide")
st.header("Impedance Matcher")

# select or upload a file


upload = st.checkbox("Upload files")

if upload:
    base = st.file_uploader("Upload a baseline signature", type=["csv"])
    sub = st.file_uploader("Upload a subsequent signature", type=["csv"])
else:
    dataDir = os.path.join(os.path.dirname(os.getcwd()), "data")
    base = st.selectbox("Select a baseline measurement", os.listdir(dataDir))
    sub = st.selectbox("Select a subsequent measurement", os.listdir(dataDir))

if base and sub and st.button("Compare"):
    # do comparison
    with st.spinner("comparing..."):
        s = timeit.timeit()
        pb, ps, pt, pa, ib, iu, it, ia, basePks, baseProm, subPks, subProm = compare(base, sub)
        e = timeit.timeit()
    st.write(f"done in {s-e}s")
    if pt == pa:
        st.success("Peak Match")
        st.write(f"peak base data: {pb}")
        st.write(f"peak sub data: {ps}")
        st.write(f"peak target key: {pt}")
        st.write(f"peak actual key: {pa}")
        st.write(f"peak base raw : {basePks}") # frequencies
        st.write(f"peak base prom : {baseProm}") # prominences
        st.write(f"peak sub raw : {subPks}") # frequencies
        st.write(f"peak sub prom : {subProm}") # prominences
    else:
        st.error("Peak mismatch")
        st.write(f"peak base data: {pb}")
        st.write(f"peak sub data: {ps}")
        st.write(f"peak target key: {pt}")
        st.write(f"peak actual key: {pa}")
        st.write(f"peak base raw : {basePks}") # frequencies
        st.write(f"peak base prom : {baseProm}") # prominences
        st.write(f"peak sub raw : {subPks}") # frequencies
        st.write(f"peak sub prom : {subProm}") # prominences
    if it == ia:
        st.success("Impedance Match")
        st.write(f"impedance base data: {ib}")
        st.write(f"impedance sub data: {iu}")
        st.write(f"impedance target key: {it}")
        st.write(f"impedance actual key: {ia}")
    else:
        st.error("Impedance mismatch")
        st.write(f"impedance base data: {ib}")
        st.write(f"impedance sub data: {iu}")
        st.write(f"peak target key: {it}")
        st.write(f"peak actual key: {ia}")

    