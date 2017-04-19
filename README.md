# ucl-irdm-2017 Project 2 group 29

Requirments
* numpy
* scikit learn
* pickle
* Tensorflow 0.12

Python Files
* logreg.py - This file handles the logistic regression classifier. It is ran using fold1, and requires that the data is in a folder named Fold1 within the same directoy. It will save a list of NDCG@10 and ERR scores for each queryid which can then be used by signif_test.py in order to emply test statistics
* baseline.py - This file handles the baseline model. It is ran using fold1, and requires that the data is in a folder named Fold1 within the same directoy. It will save a list of NDCG@10 and ERR scores for each queryid which can then be used by signif_test.py in order to emply test statistics
* RankNet.py - This file handles the implementation of RankNet. It is run by using sets S1-4 from the MSLR-10k dataset and using S5 as the test set. To run, download the data set and extract the subsets S1-5 which are labeled on the Microsoft webpage. Put into a folder labeled MSLR-10k. It will save a list of NDCG and ERR values of the last test epoch and can be used to test significance.

Pickle Files - These are the files that are used by signif_test.py.
Numpy Files -  These are the files that are used by signif_test.py.
