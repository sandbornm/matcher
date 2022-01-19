# Copyright (c) 2021 Michael Sandborn (michael.sandborn@vanderbilt.edu)

import argparse
import os
from re import L
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import floor, log2
from itertools import combinations
from tqdm import tqdm


"""
Todo:

    support additional data formats
    other command line args
    improve error handling
    clean up data directory management
"""

TEST_DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "match", "test")
TEST_DATA_FILES = sorted(os.listdir(TEST_DATA_DIR))

VT_DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "match", "VT")
VT_DATA_FILES = sorted(os.listdir(VT_DATA_DIR))

TT_DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), "match", "TT")
TT_DATA_FILES = sorted(os.listdir(TT_DATA_DIR))
TT_HEADER_SIZE = 8

XLS_HEADER_SIZE = 14

ALL_FILES = VT_DATA_FILES + TT_DATA_FILES

MULTI_DIR = os.path.join(os.path.dirname(os.getcwd()), "match", "multi")
MULTI_FILES = sorted(os.listdir(MULTI_DIR))

ORIG_DIR = "/Users/michael/Downloads/allParts"

CUTOFF = 200
MAX_CUTOFF = 4000

PARTS = ["QR", "UT", "Tx"]
TEST_PARTS = ["A"]
VU_PARTS = ["BOX", "BRK", "FLG", "IMP", "VNT"]

GROUPS = ["all", "test", "orig", "multi"]



def impedanceToInt(data):
    return np.rint(data)

def getBitCount(n):
    return int(floor(log2(n)) + 1)

def bitWidth(w1, w2):
    return max(w1, w2)

def getHammingDistance(s1, s2):
    assert len(s1) == len(s2), f"length mismatch: len(s1)={len(s1)}, len(s2)={len(s2)}"
    hd = sum(c1 != c2 for c1, c2 in zip(s1, s2))
    # report % match instead of raw hamming distance in bits because of different encoding lengths
    pmatch = round((len(s1)-hd)/len(s1) * 100, 3)
    return (hd, pmatch)

def getPairs(n):
    #print(f"n is {n}")
    return list(combinations(range(n), 2))

def interp(x, xp, fp):
    return np.interp(x, xp, fp)

def alignSignature(i1, i2, f1, f2, l):
    if len(i1) < len(i2): # shorten to i1 len
        ia2 = i2[:l]
        ia1 = interp(f2[:l], f1[:l], i1[:l])
    else: # shorten to i2 len
        ia1 = i1[:l]
        ia2 = interp(f1[:l], f2[:l], i2[:l])
    assert len(ia1) == len(ia2)
    return (ia1, ia2)


"""
    Get the label of the passed file f given the set of parts
    of interest. The label is a tuple of the form (id, idn, orig)
    where id is the part name, idn is the part instance number, and
    orig is the original source of the data.

"""
def getLabel(parts, f):
    # very hacky, but works for now
    sid = f.split("_")
    isNpy = f[-3:] == "npy"
    isXls = f[-3:] == "xls"
    #print(sid)
    # idx = f.index("_")
    # idx2 = f[idx+1:].index("_")
    # id = f[:idx]

    id = sid[0]
    if "QR" in id:
        id = "QR"

    #print(f"id is {id}")
    #if len(idn) == 2:
    if not isNpy:
        idn = sid[1][:sid[-1].index(".")]
    else:
        idn = sid[1] #+ "_" + sid[-1][0]
        sc = sid[-1][0]

    if "A" in parts:
        orig = "AD1" if f[-7:-4] == "AD1" else "AD2"
    else:
        if f[-3:] == "csv":
            orig = "vt"
        elif f[-3:] == "txt":
            orig = "tt"
        elif isNpy or isXls:
            orig = "vu"
    # print(f"id is {id}")
    # print(f"idn is {idn}")
    # print(f"sc is {sc}")

    #print(f"id: {id} idn: {idn} orig: {orig}")
    if isNpy:
        return (id, idn, orig, sc)
    else:
        return (id, idn, orig, None)


def readNpy(fname):
    d = np.load(fname)
    frq = d[:, 0]
    imp = d[:, 1]
    return frq, imp


"""
Read the impedance data as a .csv file or a .txt file

.csv expected format:
    frequency, real impedance, imaginary impedance (3 columns, no header)

    todo: add other supported formats
"""
def readData(filename, dir=None):
    if filename[-3:] == "csv":
        try:
            if dir is None: # filename is absolute path
                df = pd.read_csv(filename, header=None)
            else:
                df = pd.read_csv(os.path.join(dir, filename), header=None)
            frq = np.array(df.loc[:, 0])
            imp = np.array(df.loc[:, 1])
            width = getBitCount(max(imp))
        except ValueError:
            print("invalid file format! Exiting...")

        return (frq, imp, width)
    elif filename[-3:] == "txt":
        try:
            if dir is None:
                with open(filename, "r") as f:
                    lns = f.readlines()
            else:
                fn = os.path.join(dir, filename)
                with open(fn) as f:
                    lns = f.readlines()
            frq = []
            imp = []
            for ln in lns[TT_HEADER_SIZE:]:
                l = ln.strip().split(";")
                frq.append(float(l[0].strip()))
                imp.append(float(l[-1].strip()))
            frq = np.asarray(frq)
            imp = np.asarray(imp)
            width = getBitCount(max(imp))
            return (frq, imp, width)
        except ValueError:
            print("invalid file format! Exiting...")
    elif filename[-3:] == "npy":
        if dir is None:
            frq, imp = readNpy(fname)
        else:
            frq, imp = readNpy(os.path.join(dir, filename))
        width = getBitCount(max(imp))
        return (frq, imp, width)
    elif filename[-3:] == "xls":
        try:
            if dir is None:
                df = pd.read_excel(filename, header=None)
            else:
                df = pd.read_excel(os.path.join(dir, filename), header=None)
            df = df[XLS_HEADER_SIZE:]
            frq = np.array(df.loc[:, 0], dtype="float64")
            imp = np.array(df.loc[:, 1], dtype="float64")
            print(frq[:10])
            print(imp[:10])
            width = getBitCount(max(imp))
            return (frq, imp, width)
        except ValueError:
            print("invalid file format! Exiting...")
    else:
        raise ValueError("invalid file type (expected csv or txt). Exiting...")

"""
    get the directories for the files in the passed list of files

    assumes the files array has relative paths (not needed for abspaths)
"""
def getDirs(fl):
    drs = [None for f in fl]
    for i, f in enumerate(fl):
        if "A" in f:
            drs[i] = TEST_DATA_DIR
        elif f[-3:] == "csv":
            drs[i] = VT_DATA_DIR
        else:
            drs[i] = TT_DATA_DIR
    return drs

"""
Converts a single impedance value to binary with
specified width
"""
def impedanceToBinary(val, wdth):
    return np.binary_repr(val.astype(int), width=wdth)


"""
Convert the entire impedance signature to binary with
pointwise conversion from integer response to bits
"""
def signatureToBinary(data, wdth):
    st = ""
    intImp = []
    for d in data:
        st += impedanceToBinary(d, wdth)
        intImp.append(d.astype(int))
    return st, intImp


"""
    Create the histograms of data from given the data and list of parts:

    data: dictionary of part instance pairs : % match between the parts given in the key
    parts: list of parts of interest

"""
def makeHistograms(data, parts, l):

    print(f"len data {len(data)}")

    fig, axs = plt.subplots(len(data), 2, sharey=True, tight_layout=True)
    print(f"ax len {axs.shape}")

    if "A" in parts: # histogram for test data
        for i, p in enumerate(parts):
            d1 = data[p][0] # same type
            d2 = data[p][1] # same instance
            axs[0].hist(d1.values(), bins=5)
            axs[1].hist(d2.values(), bins=5)
            axs[0].set_title(f"{p} Inter (n={len(data[p][0])})")
            axs[1].set_title(f"{p} Intra (m={len(data[p][1])})")

            colrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            for j, d in enumerate(list(d2.values())[:3]):
                axs[0].axvline(x=d, color=colrs[j], linestyle='dashed', linewidth=2, label=list(d2.keys())[j])
                axs[0].legend(loc="upper right", prop={'size': 6})
        axs[0].set_xlabel("Bitwise % match")
        axs[0].set_ylabel("Count")

    else:
        for i, p in enumerate(parts):
            d1 = data[p][0]
            d2 = data[p][1]
            axs[i][0].hist(d1.values(), bins=5)
            axs[i][1].hist(d2.values(), bins=5)
            axs[i][0].set_title(f"{p} Inter (n={len(data[p][0])})")
            axs[i][1].set_title(f"{p} Intra (m={len(data[p][1])})")

            colrs = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            for j, d in enumerate(list(d2.values())[:3]):
                axs[i][0].axvline(x=d, color=colrs[j], linestyle='dashed', linewidth=2, label=list(d2.keys())[j])
                axs[i][0].legend(loc="upper right", prop={'size': 4})
        axs[0][0].set_xlabel("Bitwise % match")
        axs[0][0].set_ylabel("Count")
    plt.suptitle(f"% Match of Pairwise Binary Impedance Signature \n of the Same Part Type and Instance on Different Analyzers \n (length: {l} responses)", fontsize=8)
    plt.show()


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", help="abs path of first file to read",
                        type=str)
    parser.add_argument("-f2", help="abs path of second file to read",
                        type=str)
    parser.add_argument("-group", help="the group of data ('test' or 'all')",
                        type=str)
    parser.add_argument("-l", help="the number of response values to encode", type=int, default=CUTOFF)

    args = parser.parse_args()
    print(f"args {args}")

    if args.group is not None:
        assert args.group in ["test", "all"], f"invalid group: {args.group}"
    grp = args.group # test
    if grp == "test":
        dir = TEST_DATA_DIR
        fls = TEST_DATA_FILES
    elif grp == "vt":
        dir = VT_DATA_DIR
        fls = VT_DATA_FILES
    elif grp == "tt": # tt
        dir = TT_DATA_DIR
        fls = TT_DATA_FILES
    else: # grp == "inter"
        dir = None
        fls = ALL_FILES
    parts = TEST_PARTS if args.group == "test" else PARTS

    return (args, dir, fls, parts)


def run(dir, fls, parts, l):
    print(fls)
    data = {p : [] for p in parts}
    for part in parts:
        print(f"part {part}")
        # filter files
        fl = [f for f in fls if part in f]
        prs = getPairs(len(fl))

        typ = {}
        inst = {}

        for p in tqdm(prs):
            fn1 = fl[p[0]]
            fn2 = fl[p[1]]

            la1, id1, org1, sc1 = getLabel(parts, fn1)
            la2, id2, org2, sc2 = getLabel(parts, fn2)

            #print(la1, id1, org1)
            #print(la2, id2, org2)

            if dir is None: # all files
                d1 = VT_DATA_DIR if fn1[-3:] == "csv" else TT_DATA_DIR
                d2 = VT_DATA_DIR if fn2[-3:] == "csv" else TT_DATA_DIR
                f1, i1, w1 = readData(fn1, d1)
                f2, i2, w2 = readData(fn2, d2)
            else:
                f1, i1, w1 = readData(fn1, dir)
                f2, i2, w2 = readData(fn2, dir)

            wid = bitWidth(w1, w2)

            ia1, ia2 = alignSignature(i1, i2, f1, f2, l)
            s1, simp1 = signatureToBinary(ia1, wid)
            s2, simp2 = signatureToBinary(ia2, wid)

            hd, pmatch = getHammingDistance(s1, s2)

            if ".npy" in fls[1]: # add scan number to prevent dictionary overwrites
                if id1 == id2: # same part instance
                    #print(f"same instance {id1} and {id2}")
                    inst[f"{la1}_{id1}_{sc1}, {la2}_{id2}_{sc2}"] = pmatch
                else: # same part type
                    #print(f"same type {la1} and {la2}")
                    typ[f"{la1}_{id1}_{sc1}, {la2}_{id2}_{sc2}"] = pmatch

            else:
                if id1 == id2: # same part instance
                    #print(f"same instance {id1} and {id2}")
                    inst[f"{la1}_{id1}_{org1}, {la2}_{id2}_{org2}"] = pmatch
                else: # same part type
                    #print(f"same type {la1} and {la2}")
                    typ[f"{la1}_{id1}_{org1}, {la2}_{id2}_{org2}"] = pmatch

        data[part].append(typ)
        data[part].append(inst)

        #print(f"# type pairs for {part}: {len(data[part][0].values())}") # same type
        #print(f"# instance pairs for {part}: {len(data[part][1].values())}") # same instance

    return data

def getDirectory(group):
    print("group test")
    assert group in GROUPS, f"invalid group {group}"
    if group == "all":
        dir = None
        fls = ALL_FILES
    elif group == "test":
        dir = TEST_DATA_DIR
        fls = TEST_DATA_FILES
    elif group == "multi":
        dir = MULTI_DIR
        fls = MULTI_FILES
    elif group == "orig":
        dir = ORIG_DIR
        fls = sorted(os.listdir(dir))
    if dir == TEST_DATA_DIR:
        parts = TEST_PARTS
    elif dir == ORIG_DIR:
        parts = VU_PARTS
    else:
        parts = PRTS
    return (dir, fls, parts)


# # driver
if __name__ == "__main__":
    group = "orig"
    dir, fls, parts = getDirectory(group)
    data = run(dir, fls, parts, CUTOFF)

    # for f in os.listdir(pdi):
    #     fn = f[:3] + "_" + f[3:]
    #     print(f)
    #     print(fn)
    #     os.rename(os.path.join(pdi, f), os.path.join(pdi, fn))
    # print(os.listdir(pdi))
#     #args, dir, fls, parts = getArgs()
#     group = "test"
#     dir, fls, parts = getDirectory(group)
#     data = run(dir, fls, parts, CUTOFF)
#     print(data)
#     makeHistograms(data, parts, CUTOFF)

    # for f in os.listdir(dir):
    #     if f.startswith("UT"):
    #         print(f)
    #         fn = "UT_" + f[2:]
    #         print(fn)
    #         os.rename(os.path.join(dir, f), os.path.join(dir, fn))



    #assert args.l > 0 and args.l <= MAX_CUTOFF, f"given length {args.l} but max length is {MAX_CUTOFF}"
    # if args.f1 is not None and args.f2 is not None:
    #     f1 = args.f1
    #     f2 = args.f2
    #
    #     # check relative paths
    #     if "/" not in f1 or "/" not in f2:
    #         raise ValueError(f"Expected absolute paths. Invalid paths: {f1} {f2}")
    #         # fl = [f1, f2]
    #         # d = getDirs(fl)
    #     else:
    #         f1, i1, w1 = readData(f1)
    #         f2, i2, w2 = readData(f2)
    #         ia1, ia2 = alignSignature(i1, i2, f1, f2, args.l)
    #         wid = bitWidth(w1, w2)
    #         s1, simp1 = signatureToBinary(ia1, wid)
    #         s2, simp2 = signatureToBinary(ia2, wid)
    #         hd, pmatch = getHammingDistance(s1, s2)
    #         print("###############")
    #         print(f"Comparing {args.f1} and {args.f2}")
    #         print(f"Encoding length: {wid} bits")
    #         print(f"Hamming distance: {hd}/{len(s1)} bits = {pmatch}% match")
    #         print("###############")
    # elif args.group is not None: # valid group, no files
    #     data = run(dir, fls, parts, args.l)
    #     makeHistograms(data, parts, args.l)
    # else:
    #     raise ValueError ("Expected two files to compare or a group")
