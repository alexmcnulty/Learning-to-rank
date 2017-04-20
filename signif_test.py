
# coding: utf-8

# In[ ]:

import pickle
import numpy as np
from scipy import stats

with open('base_metrics.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    ndcg_base, err_base = pickle.load(f)

with open('logreg_metrics.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    ndcg_log, err_log = pickle.load(f)



############ Logistic
print('Logistic results:')
print(stats.wilcoxon(np.asarray(ndcg_base),np.asarray(ndcg_log)))

print(stats.wilcoxon(np.asarray(err_base),np.asarray(err_log)))


print(stats.ttest_ind(np.asarray(ndcg_base),np.asarray(ndcg_log),equal_var=False))

print(stats.ttest_ind(np.asarray(err_base),np.asarray(err_log),equal_var=False))

############ RankNet
filename1= 'RN_ndcg_list_test.npy'
filename2 = 'RN_err_list_test.npy'
ndcg_RN = np.load(filename1)
err_RN = np.load(filename2)

print('\nRankNet results:')
print(stats.wilcoxon(np.asarray(ndcg_base),np.asarray(ndcg_RN)))

print(stats.wilcoxon(np.asarray(err_base),np.asarray(err_RN)))


print(stats.ttest_ind(np.asarray(ndcg_base),np.asarray(ndcg_RN),equal_var=False))

print(stats.ttest_ind(np.asarray(err_base),np.asarray(err_RN),equal_var=False))

############ LambdaRank
filename1= 'LR_ndcg_list_test.npy'
filename2 = 'LR_err_list_test.npy'
ndcg_LR = np.load(filename1)
err_LR = np.load(filename2)

print('\nLambdaRank results:')
print(stats.wilcoxon(np.asarray(ndcg_base),np.asarray(ndcg_LR)))

print(stats.wilcoxon(np.asarray(err_base),np.asarray(err_LR)))


print(stats.ttest_ind(np.asarray(ndcg_base),np.asarray(ndcg_LR),equal_var=False))

print(stats.ttest_ind(np.asarray(err_base),np.asarray(err_LR),equal_var=False))




############ LambdaMART
filename3= 'LM_ndcg_list_test.npy'
filename4 = 'LM_err_list_test.npy'
ndcg_LM = np.load(filename3)
err_LM = np.load(filename4)


print('\nLambdaMart results:')
print(stats.wilcoxon(np.asarray(ndcg_base),np.asarray(ndcg_LM)))

print(stats.wilcoxon(np.asarray(err_base),np.asarray(err_LM)))

print(stats.ttest_ind(np.asarray(ndcg_base),np.asarray(ndcg_LM),equal_var=False))

print(stats.ttest_ind(np.asarray(err_base),np.asarray(err_LM),equal_var=False))


########### LambdaMart and Logistic

print('\nSignificane between LambdaMart and Logistic regression:')
print(stats.wilcoxon(np.asarray(ndcg_log),np.asarray(ndcg_LM)))

print(stats.wilcoxon(np.asarray(err_log),np.asarray(err_LM)))


print(stats.ttest_ind(np.asarray(ndcg_log),np.asarray(ndcg_LM),equal_var=False))

print(stats.ttest_ind(np.asarray(err_log),np.asarray(err_LM),equal_var=False))


