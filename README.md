# ucl-irdm-2017 Project 2 group 29

Requirments
* numpy
* scikit learn
* pickle
* Tensorflow 0.12

Python Files
* logreg.py - This file handles the logistic regression classifier. It is ran using fold1, and requires that the data is in a folder named Fold1 within the same directoy. It will save a list of NDCG@10 and ERR scores for each queryid which can then be used by signif_test.py in order to employ test statistics
* baseline.py - This file handles the baseline model. It is ran using fold1, and requires that the data is in a folder named Fold1 within the same directoy. It will save a list of NDCG@10 and ERR scores for each queryid which can then be used by signif_test.py in order to employ test statistics
* RankNet.py - This file handles the implementation of RankNet. It is run by using sets S1-4 from the MSLR-10k dataset and using S5 as the test set. To run, download the data set and extract the subsets S1-5 which are labeled on the Microsoft webpage. Put into a folder labeled MSLR-10k. It will save a list of NDCG and ERR values of the last test epoch and can be used to test significance.
* RankNet.ipynb - This is a note book version of RankNet.py. It can be easier to read and has insturcions on how to load the final model. Put the accomanying best_accuracy files into a folder labeled models to get the result. 

* LambdaRank_NDCG.py - (incorrect implementation) after defining functions and loading data (S1-3 for training, S5 for testing), the main loop obtains lambdas per document, backpropagates to find parameter gradients, and multiplies the two together to make the weight updates. The training set is passed through the neural net for the next iteration. After the loop, the test set is passed through the neural nets, and the train/test scores are plotted.
* LambdaRank_ERR.py - (incorrect implementation) as above, but for training with the ERR metric
* LambdaMART_NDCG.py - in the main loop, lambdas and their derivatives are found, before training a tree to predict lambdas. Then, the tree's leaves are replaced with Newton-Raphson terms. The tree is appended to the ensemble, and the predicted scores adjusted accordingly. After the loop, the test set is passed through the ensemble, and the train/test scores are plotted.
* LAMBDAMART_ERR_2.py - as above, but for training with the ERR metric.



the folder named tests holds all of the data from the models, and the baseline.py which is used to run the significance tests
* Pickle Files - These are the files that are used by signif_test.py.
* Numpy Files -  These are the files that are used by signif_test.py.

best_accuracy. files - the final ranknet tensorflow model is saved here.  
