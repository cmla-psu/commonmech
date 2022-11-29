import pandas as pd
import numpy as np
df = pd.read_csv('..//datafile//brazil.csv')

age_range = np.sort(df['AGE'].unique())
sex_range = np.sort(df['SEX'].unique())
state_range = np.sort(df['GEO1_BR'].unique())
occ_range = np.sort(df['OCC'].unique())

for col in df.columns:
    col_range = len(df[col].unique())
    print(col, " ", col_range)

# df_ag = df.pivot_table(
#     index=['GEO1_BR', 'OCC', 'AGE', 'SEX'], aggfunc='size')

# size_m = len(state_range) * len(occ_range)
# size_n = len(age_range) * len(sex_range)
# data = np.zeros([size_m, size_n])
#
#
# def get_col_index(age, sex):
#     return age + 101 * (sex - 1)
#
#
# data_sum = np.sum(data, axis=1)
# idx_greater = np.where(data_sum > 0)[0]
# data_greater = data[idx_greater, :]
# # np.save("brazil.npy", data_greater)
