import os
from numpy.core.defchararray import isalnum
import pandas as pd
import numpy as np
from pandas.core import base
from scipy.signal import find_peaks
from scipy import spatial
from scipy.signal.ltisys import impulse
from fuzzyExtractor import FuzzyExtractor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from perlin_noise import PerlinNoise
import itertools
import altair as alt
import matplotlib.pyplot as plt


import plotly.express as px

THRESHOLD = 15


class Part:
    def __init__(self, file=None, real=None, imag=None, freq=None, name=None, length=5, tolerance=5):
        if isinstance(file, str):
            assert file[-4:] == ".csv", "CSV file expected" 
            self.name = file
            self.loadData()
            print(f"self.name {self.name} \n self.realImp {self.realImp}")
        elif real is not None and freq is not None and name is not None:
            # create a part instance from noisy data from a baseline
            self.name = name
            self.freq = freq
            self.realImp = real
            print(f"self.name {self.name} \n self.realImp {self.realImp}")
            self.imagImp = imag # use the same imaginary data as the baseline for now

        else: # file upload
            if name is not None:
                self.name = name
            file.seek(0) # weird hack to set file position for pandas to detect data
            self.loadData(upload=True, data=file)
        
        self.getPeaks()
        self.normData()
        self.getBinary()
        # self.length = length
        # self.tolerance = tolerance
        # self.peakBytes = peaksToByteArray(self.peaks)
        # self.impBytes = self.packImpedance()

        # self.diffstring = self.getDiffString()

        # peaks
        #print("Fe peaks")
        # self.fe = FuzzyExtractor(length, tolerance)
        # self.k, self.h = self.fe.generate(self.peakBytes[:length])

        # impedance
        #print("Fe impedance")
        # self.fei = FuzzyExtractor(length, tolerance)
        # self.ki, self.hi = self.fe.generate(self.impBytes[:length])


        # self.splits = len(self.freq) // 4
        # self.split1 = self.realImp[:self.splits]
        # self.split2 = self.realImp[self.splits:self.splits*2]
        # self.split3 = self.realImp[self.splits*2:self.splits*3]
        # self.split4 = self.realImp[self.splits*3:]
        # self.snrs = [self.snr(s) for s in [self.split1, self.split2, self.split3, self.split4]] 
        # self.snrRaw = self.snr(self.realImp)
        # self.snrScale = self.snr(self.normReal)

    def loadData(self, upload=False, data=None):
        if upload:
            df = pd.read_csv(data, header=None) # pandas reads file as buffer
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "data", self.name), header=None)
        
        assert len(df.columns) == 3, "bad file format, expected 3 columns"
        self.freq = np.array(df.loc[:, 0])
        self.realImp = np.array(df.loc[:, 1])
        self.imagImp = np.array(df.loc[:, 2])

    def __repr__(self):
        s = f"Name: {self.name.split('/')[-1]}\n"
        s += f"Range: {min(self.freq)}-{max(self.freq)}Hz \n"
        s += f"Size: {len(self.freq)}points"
        return s

    def getPeaks(self, numPeaks=10):
        pks, props = find_peaks(self.realImp, prominence=1)
        peakIdx = np.flip(np.argsort(props['prominences']))[:numPeaks] # this is sorted peak prominence in descending order
        self.peaks = pks[peakIdx]
        self.prom = props['prominences'][peakIdx]
        self.peakFrq = self.freq[self.peaks]

        print(f"peak prominence {self.prom}")
        print(f"peak frequency {self.peakFrq}")

    def normData(self):
        real = self.realImp
        imag = self.imagImp
        self.normReal = (real - np.min(real)) / (np.max(real) - np.min(real))
        self.normImag = (imag - np.min(imag)) / (np.max(imag) - np.min(imag))
        self.normFreq = (self.freq - np.min(self.freq)) / (np.max(self.freq) - np.min(self.freq))

    def packImpedance(self):
        #print("packing")
        #print(self.normReal[:self.tolerance])
        # todo numpy tobytes() vs struct pack
        bytes = self.normReal[:self.tolerance].tobytes()
        #print("packed impedance")
        #print(bytes)
        return bytes

    """ signal to noise ratio """
    def snr(self, data, axis=0, ddof=0):
        mu = data.mean(axis=axis)
        sig = data.std(axis=axis, ddof=ddof)
        return np.where(sig == 0, 0, mu/sig)

    """ create an image from the signal data"""
    def makeImgData(self):
        # todo play around with this 
        imd = np.outer(self.normReal, self.freq)
        return np.round(np.mod(imd, 256))
    
    """ constructs a binary string based on whether 
    the last response values was larger than (1) or smaller than (0)
    the previous value
    """
    def getDiffString(self):
        ds = ""
        for i in range(len(self.realImp) - 1):
            if self.realImp[i] > self.realImp[i+1]:
                ds += "1"
            else:
                ds += "0"
        return ds


    """ make the impedance data into a binary string by taking the first digit past the decimal
    of the normalized impedance values and converting to binary"""
    def getBinary(self):
        bmap = {0: "0000",
                1: "0001",
                2: "0010",
                3: "0011",
                4: "0100",
                5: "0101",
                6: "0110",
                7: "0111",
                8: "1000",
                9: "1001",
                }
        s = ""
        for x in self.normReal:
            if x == 1:
                s+= bmap[1]
            else: 
                y = x * 10
                z = int(y)
                s+= bmap[z]
        b = bin(int(s, 2))
        self.binary = b # this is the binary form of all tenths place normalized impedance values


""" compare 2 parts by specifying a baseline and retrieving
    fuzzy extractor data

"""
class Comparator:
    def __init__(self, part1, part2, baseline='1'):
        assert isinstance(part1, Part)
        assert isinstance(part2, Part)
        assert baseline in ('1', '2')
        # init parts to compare
        self.base, self.sub = (part1, part2) if baseline == '1' else (part2, part1)

        # # keep peaks byte data 
        # self.peakBase = self.base.peakBytes[:self.base.length]
        # self.peakSub = self.sub.peakBytes[:self.base.length]
        
        # # keys for peaks
        # self.peakTargetKey, self.peakHelpers = self.base.k, self.base.h
        # self.peakActualKey = self.base.fe.reproduce(self.peakSub, self.peakHelpers)

        # # keep normalized impedance byte data 
        # self.impBase = self.base.impBytes[:self.base.tolerance]
        # self.impSub = self.sub.impBytes[:self.base.tolerance]

        # # keys for impedance
        # self.impTargetKey, self.impHelpers = self.base.ki, self.base.hi
        # self.impActualKey = self.base.fei.reproduce(self.impSub, self.impHelpers)

        # get metrics
        self.metrics = self.getMetrics()

    def getHammingDistance(self):
        baseBin = self.base.binary
        subBin = self.sub.binary
        assert len(baseBin) == len(subBin), "length mismatch"

        cnt = 0
        for i in range(len(baseBin)):
            if baseBin[i] != subBin[i]:
                cnt += 1
        # cnt is number of bit differences, len(baseBin) is total number of bits
        return (cnt, len(baseBin))

    def getMetrics(self):
        names = ["mse", "rmse", "l1", "hamming", "cosine", "tolString"] # add others
        mse = mean_squared_error(self.base.normReal, self.sub.normReal)
        rmse = np.sqrt(mse)
        l1 = self.base.normReal - self.sub.normReal
        ham = self.getHammingDistance()
        cos = 1 - spatial.distance.cosine(self.base.normReal, self.sub.normReal)
        # binary string based on tolerances
        # todo plot differences wrt to frequency - increasing/decreasing?
        # todo provide other constructions of binary sequences
        lb = self.base.normReal - .05 # todo change these based on known variation
        ub = self.base.normReal + .05
        l1s = ["0" if x >= lb[i] and x <= ub[i] else "1" for i,x in enumerate(self.sub.normReal)]
        l1s = "".join(l1s)

        return dict(zip(names, [mse, rmse, l1, ham, cos, l1s]))   



""" Standalone helpers """


def peaksToByteArray(peaks):
    # todo check default to floor, output of peaks are floats 
    return b''.join([int(x).to_bytes(2, byteorder='big') for x in peaks])

    
def compare(base, sub):
    print(f"not isinstance(base, list): {not isinstance(base, list)}")
    if not isinstance(base, list):
        basePart = Part(base, name=base.name[:-4])
        subPart = Part(sub, name=sub.name[:-4])
    else:
        basePart = Part(base)
        subPart = Part(sub)
    cmp = Comparator(basePart, subPart)
    # todo add viz - 
    # todo perturb the baseline to "damage" change every 100 values by some noise
    # todo check byte symbols in python peak sub data: b'\x00\x13\x00F\x00' from AD2_DitherA5.csv

    return cmp, basePart, subPart

""" make interactive plotly charts instead of matplotlib static charts"""
def plotly(base, sub):
    basePart = Part(base)
    subPart = Part(sub)

    fig = go.Figure({'layout': {'title' : {'text' : 'Base vs. Sub (normalized)'} } })
    
    # base
    fig.add_trace(go.Scatter(
        x=basePart.freq,
        y=basePart.normReal,
        mode="lines", 
        name="base signature"
    ))

    # sub
    fig.add_trace(go.Scatter(
        x=subPart.freq,
        y=subPart.normReal,
        mode="lines", 
        name="sub signature"
    ))

    return fig


""" generate data from a base measurement with Perlin noise with predefined octaves and scales"""
def generate(selections, numSamples):

    parts = []
    for x in selections:
        p = Part(x)
        parts.append(p)
    # handle case of no noise data
    if numSamples == 0:
        return parts, None, None, None

    #noisyData = []
    # noise1 = PerlinNoise(octaves=8, seed=1)
    # noise2 = PerlinNoise(octaves=10, seed=2)
    # noise3 = PerlinNoise(octaves=12, seed=3)
    # todo add pairwise metrics for these

    #noisyData = {p.name[:-4] : [] for p in parts}
    noisyDataOrig = {p.name[:-4] : [] for p in parts}
    for p in parts:
        for i in range(numSamples):
            noise1 = PerlinNoise(octaves=8, seed=1)
            noise2 = PerlinNoise(octaves=10, seed=2)
            noise3 = PerlinNoise(octaves=12, seed=3)
            lf = len(p.freq)
            n = np.array([4 * np.random.random_sample() * noise1([i/lf]) for i in range(lf)])
            n2 = np.array([2 * np.random.random_sample() * noise2([i/lf]) for i in range(lf)])
            n3 = np.array([np.random.random_sample() * noise3([i/lf]) for i in range(lf)])
            noisyDataOrig[p.name[:-4]].append(p.realImp + n + n2 + n3)
            #noisyDataOrig[p.name[:-4]].append(p.realImp + n + n2 + n3)
    pl = plotNoise(parts, noisyDataOrig, numSamples)

    return parts, None, pl, noisyDataOrig

""" create a plotly chart of the baselines with added noise

    prts: a list of Part objects to be used as baselines in the noise generation
    noisyData: a dictionary of baseline data file name to generated noise values
 """
def plotNoise(prts, noisyData, numSamples):
    fig = go.Figure({'layout': {'title' : {'text' : f' {len(prts)} baselines with {numSamples} samples of Perlin noise (8,10,12) for each'} } })

    freq = prts[0].freq # assumes that all measurements are over the same range
    
    # baselines
    for p in prts:
        # plot baseline raw data
        fig.add_trace(go.Scatter(
            x=freq,
            y=p.realImp,
            mode="lines", 
            name=p.name[:-4]
        ))

        # noise samples for baseline
        for i,n in enumerate(noisyData[p.name[:-4]]):
            fig.add_trace(go.Scatter(
                x=freq,
                y=n,
                mode="lines",
                name=f"noise_{p.name[:-4]}_{i}"
            ))

    fig.update_layout(xaxis_title="Frequency (Hz)", yaxis_title="Real Impedance (Ohms)")

    return fig

""" string hamming distance """
def hamDist(s1, s2):
    return sum(ch1 != ch2 for ch1, ch2 in list(zip(s1, s2)))

""" Create a table of pairwise metrics for the passed data

parts: the baseline parts to be compared

"""
def createPairwiseTable(parts, noise):

    # first create all parts of interest using a shared frequency range from the baseline parts
    frq = parts[0].freq

    if noise is not None: # ie noise samples > 0
        noiseParts = []
        for i,p in enumerate(parts):
            nme = p.name[:-4]
            n = noise[nme]
            for i,x in enumerate(n):
                noiseParts.append(Part(real=x, imag=p.imagImp, freq=frq, name=f"{nme}_noise_{i}"))
        data = noiseParts + parts
    else:
        data = parts
    # collect part names and data values
    dataNames = sorted([d.name for d in data])
    #dataLookup = {d.name : (np.sort(d.peakFrq), np.sort(d.prom)) for d in data}
    dataLookup = {d.name : (d.peakFrq, d.prom) for d in data}
    dataPairs = sorted(list(itertools.combinations(dataNames, 2)))

    tblData = []
    for (d1, d2) in dataPairs:
        #print(f"d1, d2 {d1, d2}")
        pf1, pr1 = dataLookup[d1]
        pf2, pr2 = dataLookup[d2]
        #print(f"pf1 {pf1}, pr1 {pr1}")
        #print(f"pf2 {pf2}, pr2 {pr2}")

        peakFrqL1 = np.linalg.norm((pf1 - pf2), ord=1)
        peakPromL1 = np.linalg.norm((pr1 - pr2), ord=1)
        peakMatch = sum(np.not_equal(pf1, pf2)) # strict equality for peak locations
        promDiff = np.abs(pr1 - pr2) / pr2 * 100
        print(f"std dev {np.std(promDiff)}")
        promMatch = sum(promDiff < 10) # percent difference threshold for match
        isMatch = peakMatch + promMatch >= THRESHOLD
        res = "yes" if isMatch else "no"
        #print(f"promDiff {promDiff}")
        #print(f"promMatch {promMatch}")
        tblData.append((d1, d2, peakFrqL1, peakPromL1, peakMatch, promMatch, res))
    #print(f"tblData is {tblData}")

    return tblData

def getPeakPlot(data):
    print(f" len data is {len(data)}")
    da = {"Part": [], "PeakFrq": [], "PeakProm": []}
    for d in data:
        assert isinstance(d, Part)
        #print(f"len(d.freq) {len(d.peakFrq)}")
        da["Part"].append(d.name)
        da["PeakFrq"].append(sorted(list(d.peakFrq)))
        da["PeakProm"].append(sorted(list(d.prom)))
    print(f"da is {da}")
    df = pd.DataFrame(data=da)
    plt = px.line(df, x=f"PeakFrq", y=f"PeakProm", color="Part", title="Peak Frequency vs. Prominence")
    #px.line(x=prt.peakFrq, y=prt.prom, title=f"Peak Frequency vs. Peak Prominence for {prt.name}")
    return plt

""" 
    data expects a 2 element list of the values of interest
"""
def getHistogram(data):
    data = np.array(data)
    peakMatchCount = data[:, 4]
    promMatchCount = data[:, 5]

    x0 = promMatchCount #np.random.randn(250)
    x1 = peakMatchCount #np.random.randn(500) + 1

    df =pd.DataFrame(dict(
        series=np.concatenate((["peak match"]*len(x0), ["prom match"]*len(x1))), 
        data=np.sort(np.concatenate((x0,x1)))
    ))

    plt = px.histogram(df, x="data", color="series", barmode="overlay", title=f"peak count (=) vs prom count (<10%) matches (n={len(x0)})")
    plt.update_layout(xaxis_title="count", yaxis_title="frequency")
    return plt


"""
    Return the decision matrix for the passed data
"""
def getDecisionMatrix(base, sub, match):
    fig, ax = plt.subplots()
    match = [0 if m == "no" else 1 for m in match]
    match = np.array([match, match])
    mat = ax.imshow(match, cmap='GnBu', interpolation='nearest')
    plt.yticks(range(len(sub)), sub)
    plt.xticks(range(len(base)), base)
    plt.xticks(rotation=30)
    plt.xlabel('Base Signatures')

    # this places 0 or 1 centered in the individual squares
    for x in range(len(base)):
        for y in range(len(sub)):
            ax.annotate(match[x][y], xy=(y, x), horizontalalignment='center', verticalalignment='center')
    plt.show()

    # print(f"base is {base}")
    # print(f"sub is {sub}")
    # print(f"match is {match}")

    # # Compute x^2 + y^2 across a 2D grid
    # x, y = np.meshgrid(range(0, len(match)), range(0, len(match)))
    # z = np.array([0 if m == "no" else 1 for m in match])

    # #print(f"lens {len(x.ravel()), len(y.ravel()), len(z.ravel())}")

    # # Convert this grid to columnar data expected by Altair
    # source = pd.DataFrame({'x': x.ravel(),
    #                     'y': y.ravel(),
    #                     'z': z.ravel()})

    # ch = alt.Chart(source).mark_rect().encode(
    #     x='x:O',
    #     y='y:O',
    #     color='z:Q'
    # )

    # return ch

#data = os.path.join(os.pardir, "data")
#p = Part(os.path.join(data, "AD1_DitherA2.csv"))
# print(f"snr {p.snrs}")
# print(f"snr whole {p.snrRaw}")
# print(f"snr scale {p.snrScale}")

#imda = p.makeImgData()
#print(p.normFreq)
#print(f"outer data of freq and response with shape {imda.shape} is -- {imda}")
#p2 = Part(os.path.join(data, "AD2_DitherA2.csv"))

# print(hamDist(p.diffstring, p2.diffstring))
# print(len(p.diffstring))
