import os
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.signal.ltisys import impulse
from fuzzyExtractor import FuzzyExtractor


class Part:
    def __init__(self, filename, length=5, tolerance=5):
        assert filename[-4:] == ".csv", "CSV file expected" 
        self.filename = filename
        self.loadData()
        self.getPeaks()
        self.normData()
        self.length = length
        self.tolerance = tolerance
        # print("peaks")
        # print(self.peaks)

        self.peakBytes = peaksToByteArray(self.peaks)
        self.impBytes = self.packImpedance()

        # for peaks
        print("Fe peaks")
        self.fe = FuzzyExtractor(length, tolerance)
        self.k, self.h = self.fe.generate(self.peakBytes[:length])

        # for impedance
        print("Fe impedance")
        self.fei = FuzzyExtractor(length, tolerance)
        self.ki, self.hi = self.fe.generate(self.impBytes[:length])
        
    def loadData(self):
        df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), "data", self.filename), header=None)
        self.freq = np.array(df.loc[:, 0])
        self.realImp = np.array(df.loc[:, 1])
        self.imagImp = np.array(df.loc[:, 2])
        # print(self.freq[:10])
        # print(self.realImp[:10])
        # print(self.imagImp[:10])

    def __repr__(self):
        s = f"Name: {self.filename.split('/')[-1]}\n"
        s += f"Range: {min(self.freq)}-{max(self.freq)}Hz \n"
        s += f"Size: {len(self.freq)}points"
        return s

    def getPeaks(self, numPeaks=10):
        # todo sort and select n most prominent
        # peak properties has prominence, sort by prominence
        # 
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
        #return b''.join(bytearray(byte) for byte in bytes)
    
        

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


        # print("Peaks")
        # print(f"source: {self.peakSource}")
        # print(f"input: {}")
        # print(f"Baseline key: {self.peakTarget}")
        # print(f"Subsequent key: {self.peakActual}")

        # keep normalized impedance byte data 
        self.impBase = self.base.impBytes[:self.base.tolerance]
        self.impSub = self.sub.impBytes[:self.base.tolerance]

        # keys for impedance
        self.impTargetKey, self.impHelpers = self.base.ki, self.base.hi
        self.impActualKey = self.base.fei.reproduce(self.impSub, self.impHelpers)
            
        # print("Impedance")
        # print(f"source: {base.impBytes[:base.tolerance]}")
        # print(f"input: {sub.impBytes[:base.tolerance]}")
        # print(f"Baseline key: {tar}")
        # print(f"Subsequent key: {act}")


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
    # todo add binary strings corresponding

    return (cmp.peakBase,
            cmp.peakSub,
            cmp.peakTargetKey,
            cmp.peakActualKey,
            cmp.impBase,
            cmp.impSub,
            cmp.impTargetKey, 
            cmp.impActualKey,
            cmp.base.peakFrq,
            cmp.base.prom, 
            cmp.sub.peakFrq,
            cmp.sub.prom)