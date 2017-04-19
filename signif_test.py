
# coding: utf-8

# In[14]:

import pickle
import numpy as np
from scipy import stats

with open('base_metrics.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    ndcg_base, err_base = pickle.load(f)

with open('logreg_metrics.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    ndcg_log, err_log = pickle.load(f)



print(np.shape(np.asarray(err_base)))

############ Logistic
print('Logistic results:')
print(stats.wilcoxon(np.asarray(ndcg_base),np.asarray(ndcg_log)))

print(stats.wilcoxon(np.asarray(err_base),np.asarray(err_log)))


print(stats.ttest_ind(np.asarray(ndcg_base),np.asarray(ndcg_log)))

print(stats.ttest_ind(np.asarray(err_base),np.asarray(err_log)))

# ############ RankNet
# filename1= 'RN_ndcg_list_test.npy'
# filename2 = 'RN_err_list_test.npy'
# ndcg_RN = np.load(filename1)
# err_RN = np.load(filename1)
#
# print('\nRankNet results:')
# print(stats.wilcoxon(np.asarray(ndcg_base),np.asarray(ndcg_RN)))
#
# print(stats.wilcoxon(np.asarray(err_base),np.asarray(err_RN)))
#
#
# print(stats.ttest_rel(np.asarray(ndcg_base),np.asarray(ndcg_RN)))
#
# print(stats.ttest_rel(np.asarray(err_base),np.asarray(err_RN)))





