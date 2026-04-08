import os
import numpy as np
import pandas as pd

dataset = 'DJIA'
input_folder = f'C:/Users/Gursimranjeet Singh/REGCN/data/data/VMDdata/{dataset}/'
output_folder = f'C:/Users/Gursimranjeet Singh/REGCN/data/data/VMDnor/{dataset}/'

os.makedirs(output_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if not file_name.endswith('.csv'):
        continue
    print(f"Processing {file_name}...")
    file_path = os.path.join(input_folder, file_name)

    # ✅ force float at read time
    data = pd.read_csv(file_path, header=None).values.astype(np.float64)

    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)

    normalized = (data - min_vals) / (max_vals - min_vals + 1e-8)

    final = np.vstack([normalized, min_vals, max_vals])

    out_path = os.path.join(output_folder, file_name)
    np.savetxt(out_path, final, delimiter=',', fmt='%.10f')

