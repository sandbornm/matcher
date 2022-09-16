import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

HEADER_LEN = 17

CON_DIR = os.path.join(DATA_DIR, "CON")  # container only measurements
CONLID_DIR = os.path.join(DATA_DIR, "CONLID") # container with glued lid measurements
LID_DIR = os.path.join(DATA_DIR, "LID") # lid only measurements
SEN_DIR = os.path.join(DATA_DIR, "SEN") # sensor only measurements
TUBE_DIR = os.path.join(DATA_DIR, "TUBE") # plastic tube measurements

# if you want to add data for a new part type, it has to go here
ALL_PART_TYPES = ["CON", "CONLID", "LID", "SEN", "TUBE"]