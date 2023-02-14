import pandas as pd
import glob
import numpy as np

def convert_xls_to_df(filepath):
    ext = filepath.split(".")[-1]
    print("ext is: ", ext)

    if ext == "xls":
        header_len = 13
    with open(filepath, 'rb') as f:
        df = pd.read_excel(f, skiprows=list(range(header_len)))
    return df

def convert_df_to_numpy(df):
    dfnp = df.to_numpy()[:, :2]
    print(dfnp.shape)
    return dfnp

def save_np(filename, data):
    np.save(filename, data)

def load_np(filename):
    return np.load(filename)

filelist = glob.glob('./data/*/*.xls')

for file in filelist:
    print(f"current file: {file}")
    df = convert_xls_to_df(file)
    np_data = convert_df_to_numpy(df)
    np_file = file.replace(".xls", ".npy")
    save_np(np_file, np_data)
    print(f"saved file: {np_file}")
print("done")

print(glob.glob('./data/*/*.npy'))