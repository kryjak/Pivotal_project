import pandas as pd
import os

"""
Create a smaller version of the AdvBench dataset for testing purposes.
"""

# AdvBench 'harmful behaviours'
url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'

df = pd.read_csv(url)
n_behaviours = 50
assert n_behaviours <= len(df), "n_behaviours must be less than or equal to the number of questions in AdvBench"

df_mini = df.sample(frac=1, random_state=42)[:n_behaviours]
print(f'{df_mini.head()=}')

df_train = df_mini.sample(frac=0.8, random_state=42)
df_test = df_mini.drop(df_train.index)

# Create a directory if it doesn't exist:
path_to_folder = 'data_storage'
os.makedirs(path_to_folder, exist_ok=True)

df_train.to_csv(os.path.join(path_to_folder, 'advbench_mini_train.csv'), index=True)
df_test.to_csv(os.path.join(path_to_folder, 'advbench_mini_test.csv'), index=True)
print('Finished generating AdvBench mini and saving to a file.')
