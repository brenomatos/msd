import sys
import pandas as pd

results_file = sys.argv[1]

df = pd.read_json(results_file,lines=True)
df = df.T
print(df)
df.to_csv(results_file+".csv")

