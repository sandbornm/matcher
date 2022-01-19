import streamlit as st
import os
from utils import compare, generate, plotly, createPairwiseTable, getPeakPlot, getHistogram, getDecisionMatrix
import timeit
import pandas as pd

st.set_page_config(layout="wide")
st.header("Impedance Matcher")


def listFiles():
    dataDir = os.path.join(os.path.dirname(os.getcwd()), "data")
    return [f for f in os.listdir(dataDir) if f[-4:] == ".csv"]
files = listFiles()

# logic to make the comparison based on the user choice
def doCompare(base, sub, action="upload"):
    if action == "upload":
        pl = plotly(base, sub)
        cmp, base, sub = compare(base, sub)
        #plk = getPeakPlot(base, sub)
        c1, c2 = st.beta_columns((1, 1))
        c1.plotly_chart(pl)
        #c2.plotly_chart(plk)

if st.checkbox("Upload files"):
    base = st.file_uploader("Upload a baseline measurement", type=["csv"])
    sub = st.file_uploader("Upload a subsequent measurement", type=["csv"])

    print(base)
    if base and sub and st.button("Compare"):
        doCompare(base, sub, action="upload")

elif st.checkbox("Compare multiple measurements"):
    files = st.multiselect("Select measurements to compare", files)

    if st.checkbox("Generate synthetic data for each"):
    # sigma = st.number_input('enter a sigma for stdev of noise', min_value=0.0, max_value=0.5, step=0.01)
    # st.write(f"sigma: {sigma}")
    # smaller sigma --> same instance different measurement
    # higher sigma --> different instance
        samples = st.slider("number of samples", 0, 100, 2)
        #selections = st.multiselect("Select a measurement to add noise to", files)
        
        if st.button("Compare"):
            # st.write(sigma)
            # st.write(samples)
            # todo how to smooth noise
            
            # return generated samples as new part objects?
            parts, noise, plt, noiseOrig = generate(files, samples)
            tbl = createPairwiseTable(parts, noiseOrig)
            df = pd.DataFrame(tbl, columns=["Base", "Sub", "L1 peak loc", "L1 peak prom", "Matching Peaks", "Matching Prominence", "Match"])
            
            c1, c2 = st.beta_columns((2, 1))
            c1.plotly_chart(plt)
            c2.plotly_chart(getHistogram(tbl))
            st.table(df)

            #ch = getDecisionMatrix(df["Base"], df["Sub"], df["Match"])
            #st.altair_chart(ch) # pass in last col of table (yes/no)

            # matchCount = df[df["Match"] == "yes"].count()
            # totalCount = len(df["Match"])
            # st.write(f"match/total {matchCount}/{totalCount}")

            # # todo add metrics and histogram of pairwise HD 
            # if st.button("Pairwise metrics"):
            #     st.write("pairwise metrics")
            #     pairwiseMetrics(None) # show table of metrics`


# elif st.checkbox("Compare a single baseline with a single subsequent"):
#     base = st.selectbox("Select a baseline measurement", files)
#     sub = st.selectbox("Select a subsequent measurement", files)

#     if base and sub and st.button("Compare"):
#         with st.spinner("comparing..."):
#             s = timeit.timeit()
#             pl = plotly(base, sub)
#             cmp, base, sub = compare(base, sub)
#             e = timeit.timeit()
#         st.write(f"done in {s-e}s")

#         c1, c2, c3 = st.beta_columns((1, 1, 2))
#         c3.plotly_chart(pl)

#         c1.write(f"peak base raw : {cmp.base.peakFrq}") # frequencies
#         c1.write(f"peak base prom : {cmp.base.prom}") # prominences
#         c1.write(f"peak sub raw : {cmp.sub.peakFrq}") # frequencies
#         c1.write(f"peak sub prom : {cmp.sub.prom}") # prominences
#         # c1.write(f"peak base data: {cmp.peakBase}")
#         # c1.write(f"peak sub data: {cmp.peakSub}")
#         # c1.write(f"peak target key: {cmp.peakTargetKey}")
#         # c1.write(f"peak actual key: {cmp.peakActualKey}")

#         # todo change to different conditional check
#         # if cmp.peakTargetKey == cmp.peakActualKey:
#         #     c1.success("Peak Match")
#         # else:
#         #     c1.error("Peak mismatch")

#         # c2.write(f"imp base data: {cmp.impBase}")
#         # c2.write(f"imp sub data: {cmp.impSub}")
#         # c2.write(f"imp target key: {cmp.impTargetKey}")
#         # c2.write(f"imp actual key: {cmp.impActualKey}")
            
#         # if cmp.impTargetKey == cmp.impActualKey:
#         #     c2.success("Impedance Match")
#         # else:
#         #     c2.error("Impedance mismatch")

#         for k, v in cmp.metrics.items():
#             st.write(k, v) 
        

    