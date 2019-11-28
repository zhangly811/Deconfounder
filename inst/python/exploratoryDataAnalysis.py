import numpy as np
import scipy.io as sio
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

# exploratory analysis of measurements
measChangeIndexMat = sio.mmread("dat/measChangeIndexMat.txt")
measPercent = (scipy.sparse.coo_matrix.sum(measChangeIndexMat, axis=0)/100000).reshape(-1,1)
np.mean(measPercent)
measPercentSorted = np.sort(measPercent, axis=0)
measPercentSorted[:10]
measPercentSorted[-10:]
sum(measPercentSorted>=0.1)
sum(measPercentSorted>=0.05)
sum(measPercentSorted>=0.01)
plt.hist(measPercentSorted, bins=10)
plt.xlabel('Percent of observations')
plt.ylabel('Number of measurements')
plt.show()

measPerObs = scipy.sparse.coo_matrix.sum(measChangeIndexMat, axis=1)
measPerObs.mean()
np.median(measPerObs, axis=0)
measPerObs.max()
measPerObs.min()
plt.hist(measPerObs, bins=20)
plt.xlabel('Number of measurements')
plt.ylabel('Number of Observations')
plt.show()

# exploratory analysis of medications
drugSparseMat = sio.mmread("dat/drugSparseMat.txt")
drugPerObsSum = scipy.sparse.coo_matrix.sum(drugSparseMat, axis=1)
plt.hist(drugPerObsSum, bins=70)
plt.xlabel('Number of drugs')
plt.ylabel('Number of Observations')
plt.show()

sum(drugPerObsSum<=2)/100000

drugPercent = (scipy.sparse.coo_matrix.sum(drugSparseMat, axis=0)/100000).reshape(-1,1)
sum(drugPercent>=0.01)

# Create correlation matrix
drugDenseDf = pd.DataFrame(scipy.sparse.csr_matrix.todense(drugSparseMat))
corr_matrix = drugDenseDf.corr()
lower = corr_matrix.where(np.tril(np.ones(corr_matrix.shape), k=0).astype(np.bool))
plt.figure(figsize=(15,15))
sns.heatmap(lower,
        xticklabels=corr_matrix.columns,
        yticklabels=corr_matrix.columns)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.6
to_drop = [str(column) for column in upper.columns if any(upper[column] > 0.6)]
print ("Highly correlated drugs to drop:", to_drop)
# Drop features
drugDenseMat.drop(drugDenseMat.columns[drugDenseMat.columns.isin(to_drop)], axis=1, inplace = True)
print ("X_df.shape", drugDenseMat.shape)

# sparsity of the matrix
sparsity = np.count_nonzero(drugDenseMat)/(drugDenseMat.shape[0] * drugDenseMat.shape[1])*100
print ("Sparsity of the matrix {} %".format(100-sparsity))


plt.hist(drugPercent, bins=20)
plt.xlabel('Percentage of observations')
plt.ylabel('Number of drug')
plt.show()


