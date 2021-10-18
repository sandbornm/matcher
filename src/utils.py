import os
import pandas as pd
import numpy as np
from pandas.core import base
from scipy.signal import find_peaks
from scipy.signal.ltisys import impulse
from fuzzyExtractor import FuzzyExtractor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

import plotly.express as px


class Part:
    def __init__(self, file, length=5, tolerance=5):
        if isinstance(file, str):
            assert file[-4:] == ".csv", "CSV file expected" 
            self.filename = file
            self.loadData()
        else:
            file.seek(0) # weird hack to set file position for pandas to detect data
            self.loadData(upload=True, data=file) 
        
        self.getPeaks()
        self.normData()
        self.getBinary()
        self.length = length
        self.tolerance = tolerance
        self.peakBytes = peaksToByteArray(self.peaks)
        self.impBytes = self.packImpedance()

        # peaks
        print("Fe peaks")
        self.fe = FuzzyExtractor(length, tolerance)
        self.k, self.h = self.fe.generate(self.peakBytes[:length])

        # impedance
        print("Fe impedance")
        self.fei = FuzzyExtractor(length, tolerance)
        self.ki, self.hi = self.fe.generate(self.impBytes[:length])

        
        
    def loadData(self, upload=False, data=None):
        if upload:
            df = pd.read_csv(data, header=None) # pandas reads file as buffer
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "data", self.filename), header=None)
        
        assert len(df.columns) == 3, "bad file format, expected 3 columns"
        self.freq = np.array(df.loc[:, 0])
        self.realImp = np.array(df.loc[:, 1])
        self.imagImp = np.array(df.loc[:, 2])

    def __repr__(self):
        s = f"Name: {self.filename.split('/')[-1]}\n"
        s += f"Range: {min(self.freq)}-{max(self.freq)}Hz \n"
        s += f"Size: {len(self.freq)}points"
        return s

    def getPeaks(self, numPeaks=10):
        pks, props = find_peaks(self.realImp, prominence=1)
        peakIdx = np.flip(np.argsort(props['prominences']))[:numPeaks] # this is sorted peak prominence in descending order
        self.peaks = pks[peakIdx]
        self.prom = props['prominences'][peakIdx]
        self.peakFrq = self.freq[self.peaks]

    def normData(self):
        real = self.realImp
        imag = self.imagImp
        self.normReal = (real - np.min(real)) / (np.max(real) - np.min(real))
        self.normImag = (imag - np.min(imag)) / (np.max(imag) - np.min(imag))

    def packImpedance(self):
        print("packing")
        print(self.normReal[:self.tolerance])
        # todo numpy tobytes() vs struct pack
        bytes = self.normReal[:self.tolerance].tobytes()
        print("packed impedance")
        print(bytes)
        return bytes

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

        # keep peaks byte data 
        self.peakBase = self.base.peakBytes[:self.base.length]
        self.peakSub = self.sub.peakBytes[:self.base.length]
        
        # keys for peaks
        self.peakTargetKey, self.peakHelpers = self.base.k, self.base.h
        self.peakActualKey = self.base.fe.reproduce(self.peakSub, self.peakHelpers)

        # keep normalized impedance byte data 
        self.impBase = self.base.impBytes[:self.base.tolerance]
        self.impSub = self.sub.impBytes[:self.base.tolerance]

        # keys for impedance
        self.impTargetKey, self.impHelpers = self.base.ki, self.base.hi
        self.impActualKey = self.base.fei.reproduce(self.impSub, self.impHelpers)

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
        return cnt

    def getMetrics(self):
        names = ["mse", "rmse", "l1", "hamming"] # add others
        mse = mean_squared_error(self.base.normReal, self.sub.normReal)
        rmse = np.sqrt(mse)
        l1 = self.base.normReal - self.sub.normReal
        ham = self.getHammingDistance()

        return dict(zip(names, [mse, rmse, l1, ham]))   



""" Standalone helpers """


def peaksToByteArray(peaks):
    # todo check default to floor, output of peaks are floats 
    return b''.join([int(x).to_bytes(2, byteorder='big') for x in peaks])

    
def compare(base, sub):
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


""" generate data from a base measurement and given sigma 
    to control the variance"""
def generate(file, samples, sigma):
    prt = Part(file)
    data = prt.normReal
    freq = prt.freq
    noisyData = []
    for _ in range(samples):
        noise = np.random.normal(0, sigma, len(data))
        noised = data + noise
        noisyData.append(noised)
    noisyData = np.asarray(noisyData)
    pl = plotNoise(data, noisyData, freq, sigma, samples)

    return noisyData, pl

""" create a plotly chart of the """
def plotNoise(orig, noise, freq, sigma, samples):
    fig = go.Figure({'layout': {'title' : {'text' : f'Base with noise (sigma={sigma}, n={samples})'} } })

    # base
    fig.add_trace(go.Scatter(
        x=freq,
        y=orig,
        mode="lines", 
        name="base signature"
    ))

    # noise
    for i in range(samples):
        fig.add_trace(go.Scatter(
            x=freq,
            y=noise[i],
            mode="lines", 
        ))

    return fig


""" used for noise data- convert the array of floats into a binary string"""
def toBinary(data):
    pass