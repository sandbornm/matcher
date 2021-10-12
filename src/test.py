from utils import Part, Comparator
from fuzzyExtractor import FuzzyExtractor
import os

pt = Part(os.path.join(os.path.dirname(os.getcwd()), "data", "AD1_DitherA1.csv"))
pt2 = Part(os.path.join(os.path.dirname(os.getcwd()), "data", "AD2_DitherA1.csv"))

cmp = Comparator(pt, pt2)

