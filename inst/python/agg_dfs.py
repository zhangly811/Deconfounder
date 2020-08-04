import os
from fnmatch import fnmatch
import pandas as pd
import numpy as np

filenames = []
root='/proj/sml/usr/yixinwang/multiple_causes/blessings/src/public/gene_py/res'
for path, subdirs, files in os.walk(root):
    for name in files:
        # if fnmatch(name, pattern):
        filenames.append(os.path.join(path, name)) 

dfs = [pd.read_csv(df).T for df in filenames]
dataframes = [pd.DataFrame(df.values[1:], columns=df.iloc[0]) for df in dfs]
# print(dataframes)
dataframes = [df  for df in dataframes if df.shape[0]==1]


# if len(dataframes) > 0:
df_concat = pd.concat(dataframes,axis=0)

for column in np.array(df_concat.columns):
    if column != "simset":
        df_concat[column] = pd.to_numeric(df_concat[column])

good = np.where(df_concat["def_linear_rmse_sing_mean"] < df_concat["ppca_linear_rmse_sing_mean"] )[0]



df_concat["defpcadiffsing_linearratio"] = (df_concat["def_linear_rmse_sing_mean"] - df_concat["ppca_linear_rmse_sing_mean"] )/df_concat["ppca_linear_rmse_sing_mean"] 


df_concat["defpcadiffsing_logisticratio"] = (df_concat["def_logistic_rmse_sing_mean"] - df_concat["ppca_logistic_rmse_sing_mean"] )/df_concat["ppca_logistic_rmse_sing_mean"] 

df_concat["defpcadiff_linearratio"] = (df_concat["def_linear_rmse"] - df_concat["ppca_linear_rmse"])/df_concat["ppca_linear_rmse"]

df_concat["defpcadiff_logisticratio"] = (df_concat["def_logistic_rmse"] - df_concat["ppca_logistic_rmse"])/df_concat["ppca_logistic_rmse"]

df_concat = df_concat.sort_values("defpcadiff_linearratio")    
df_concat.to_csv(root+"/allres.csv")

df_concat.sort_values("defpcadiffsing_linearratio")[["defpcadiffsing_linearratio", "nctrl_linear_rmse_sing_mean"]]

df_concat.sort_values("defpcadiffsing_logisticratio")[["defpcadiffsing_logisticratio", "nctrl_logistic_rmse_sing_mean"]]

df_concat.sort_values("defpcadiff_linearratio")[["ppca_linear_rmse","def_linear_rmse", "trivial_mse"]]

"ppca_linear_rmse", "nctrl_linear_rmse"

df_concat.sort_values("defpcadiff_logisticratio")[["defpcadiff_logisticratio", "def_logistic_rmse", "ppca_logistic_rmse", "nctrl_logistic_rmse"]]

result = df_concat.groupby(level=0).mean()



