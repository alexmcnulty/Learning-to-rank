import numpy as np
from sys import argv
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict
import matplotlib.pyplot as plt

######################################
############ PARAMETERS ##############

N = 200	# number of trees
L = 6	# number of leaves per tree   (see Hastie/Tibshirani book)
eta = float(argv[1])	# learning rate (adjusts lambdas non-linearly through rho)

max_que = 100	# DOWNSCALE: only take this many queries per fold
max_fea = 136	# DOWNSCALE: only use this many features


######################################
############# FUNCTIONS ##############

mem = Memory("./mycache")	# cache to binary for faster load
@mem.cache
def get_data(file):
	return load_svmlight_file(file,query_id=True)

def extract_data(phase,folds):	# put into useful structures
	fe = np.empty((0,max_fea))
	sc = []
	qu = np.empty((0,),dtype=np.int32)
	for i in folds:
		features,scores,queries = S[i]
		scores = [np.array(x,dtype=np.int32) for x in separate_by_query(scores,queries)[:max_que]]
		stop = sum(len(query) for query in scores)	# number of rows for max_que
		fe = np.concatenate((fe,features[:stop,:max_fea].toarray()))
		sc.extend(scores)
		qu = np.concatenate((qu,queries[:stop]))
	return fe,sc,qu,sum(len(query) for query in sc),len(sc)

def separate_by_query(scores,queries):	# from 2 lists to 1 list of lists
	scoreslist = []
	current_query = None
	for score, query in zip(scores, queries):
		if query != current_query:	# works since query info is contiguous
			scoreslist.append([])
			current_query = query
		scoreslist[-1].append(score)
	return [np.array(x,dtype=np.float32) for x in scoreslist]

def get_ERR(q, model_sc, phase, only_total=False):	# evaluation metric
	ndocs = len(model_sc)
	pos = np.argsort(-model_sc)	# gives positions of highest to lowest model scor
	rank = np.empty(ndocs,dtype=np.int32)			# which is the numpy way to
	rank[pos] = np.arange(ndocs)	# get the rank (high to low)
	R = (2**sco[phase][q][pos] - 1)/16	# 16=2**4 (max label value)
	T = 1-R

	PI = np.ones(ndocs+1)
	ERR_to_ = np.zeros(ndocs)
	for i in range(ndocs):
		ERR_to_[i] = ERR_to_[i-1] + PI[i-1]*R[i]/(i+1) # or /np.log2(i + 2)
		PI[i] = PI[i-1] * T[i]		# PI[i] = prod(T[0 to i])

	Delta = np.zeros((ndocs,ndocs))
	for x in range(ndocs):
		for y in range(x):
			i = min(rank[x],rank[y])
			j = max(rank[x],rank[y])
			Delta[x,y] = PI[i-1] * (R[i]-R[j]) / (i+1) + \
						(T[i] - T[j]) * (ERR_to_[j-1] - ERR_to_[i]) / T[i] + \
						 PI[j-1] * (R[j]-R[i]*(T[j]/T[i])) / (j+1)
			Delta[y,x] = Delta[x,y]

	return ERR_to_[-1] if only_total else (Delta,ERR_to_[-1])


######################################
################ MAIN ################

### LOAD DATA ###
S = [None]	# dummy so index isn't confusing
for i in range(5):
	S.append(get_data('./MSLR-WEB10K/S'+str(i+1)+'.txt'))
fea,sco,que,M,Q = {},{},{},{},{}
fea['trn'],sco['trn'],que['trn'],M['trn'],Q['trn'] = extract_data('trn',[1,2,3])
fea['val'],sco['val'],que['val'],M['val'],Q['val'] = extract_data('val',[4])
fea['tst'],sco['tst'],que['tst'],M['tst'],Q['tst'] = extract_data('tst',[5])
print('data loaded')

### TRAIN ###
phase='trn'
trees = []
train_output = np.random.randn(M[phase]) / 10000	# initialise with v small random scores
ERR_train = np.zeros(N+1)
for n in range(N):
	print('tree',n)
	model_sco = separate_by_query(train_output,que[phase])

	all_lambdas = np.zeros(M[phase])
	all_H = np.zeros(M[phase])
	start = 0
	for q in range(Q[phase]):
		ndocs = len(sco[phase][q])	# number of documents

		if max(sco[phase][q]) == 0:	# if no relevant documents for query
			ERR_train[n] += 1
			all_lambdas[start:start+ndocs] = 0
			all_H[start:start+ndocs] = 0
			start += ndocs
			continue

		Delta, ERR = get_ERR(q, model_sco[q], phase)
		ERR_train[n] += ERR

		lambdas = np.zeros(ndocs)
		H = np.zeros(ndocs)
		for i in range(ndocs-1):
			for j in range(i+1,ndocs):	# all pairs once only, consider order within loop
					if sco[phase][q][i] == sco[phase][q][j]:
						continue
					else:
						delta_Z = -Delta[i,j]	# change in ERR when docs i and j swapped
						UiUj = model_sco[q][i] > model_sco[q][j]
						if UiUj:
							if delta_Z > 0: # should_swap: (i down, j up)
								e = np.exp(model_sco[q][j] - model_sco[q][i])
								rho = 1/(1 + e)	# large
								lambda_ = delta_Z * rho
								lambdas[i] -= lambda_ # will push doc i down
								lambdas[j] += lambda_ # will push doc j up
							else:    # should not swap
								e = np.exp(model_sco[q][i] - model_sco[q][j])
								rho = 1/(1 + e)	# small
								lambda_ = -delta_Z * rho
								lambdas[i] += lambda_ # will push doc i up
								lambdas[j] -= lambda_ # will push doc j down
						else:
							if delta_Z > 0: # should_swap: (i up, j down)
								e = np.exp(model_sco[q][i] - model_sco[q][j])
								rho = 1/(1 + e)	# large
								lambda_ = delta_Z * rho
								lambdas[i] += lambda_ # will push doc i up
								lambdas[j] -= lambda_ # will push doc j down
							else:    # should not swap
								e = np.exp(model_sco[q][j] - model_sco[q][i])
								rho = 1/(1 + e)	# small
								lambda_ = -delta_Z * rho
								lambdas[i] -= lambda_ # will push doc i down
								lambdas[j] += lambda_ # will push doc j up

						w = lambda_*(1-rho)
						H[i] += w
						H[j] += w

		all_lambdas[start:start+ndocs] = lambdas
		all_H[start:start+ndocs] = H
		start += ndocs

	tree = DecisionTreeRegressor(max_leaf_nodes=L)
	tree.fit(fea[phase],all_lambdas) # fit by least squares a tree to predict lambdas

	leaves = tree.apply(fea[phase])	# indices of leaf nodes reached
	leafdict = OrderedDict()
	for (i,leaf) in enumerate(set(leaves)):
		leafdict[leaf] = i	# compress to [0,n]
	numer = np.zeros(i+1)
	denom = np.zeros(i+1)
	for m in range(M[phase]):
		numer[leafdict[leaves[m]]] += all_lambdas[m]
		denom[leafdict[leaves[m]]] += all_H[m]
	denom = np.where(denom,denom,1)
	gammas = np.divide(numer,denom)	# formula for adjusted leaf node values
	tree.tree_.value[list(leafdict.keys()),0,0] = gammas	# change leaf nodes
	trees.append(tree)	# add to ensemble

	train_output += eta*tree.tree_.value[leaves,0,0]	# learning step

model_sco = separate_by_query(train_output,que[phase])
for q in range(Q[phase]):	# get final ERR score
	if max(sco[phase][q]) == 0:
		ERR_train[N] += 1
		continue
	ERR_train[N] += get_ERR(q, model_sco[q], phase, only_total=True)
ERR_train /= Q[phase]


#### VALIDATE ###
#phase = 'val
#vali_output = np.random.randn(M[phase]) / 10000
#NDCG_vali = np.zeros(N+1)
#for n in range(N+1):
#	model_sco = separate_by_query(vali_output,que[phase])
#	for q in range(Q[phase]):	# get NDCG scores
#		if not IDCG10[phase][q]:
#			NDCG_vali[n] += 1
#			continue
#		NDCG_vali[n] += get_NDCG(q, model_sco[q], phase, only_total=True)
#	if n==N: break
#	vali_output += eta*trees[n].predict(fea[phase]) # prediction step
#NDCG_vali /= Q[phase]


### PREDICT ###
phase='tst'
test_output = np.random.randn(M[phase]) / 10000
ERR_test = np.zeros(N+1)
for n in range(N+1):
	model_sco = separate_by_query(test_output,que[phase])
	for q in range(Q[phase]):	# get ERR scores
		if max(sco[phase][q]) == 0:
			ERR_test[n] += 1
			continue
		ERR_test[n] += get_ERR(q, model_sco[q], phase, only_total=True)
	if n==N: break
	test_output += eta*trees[n].predict(fea[phase]) # prediction step
ERR_test /= Q[phase]

### PLOTS ###
plt.plot([i for i in range(N+1)],ERR_train,'b')
plt.plot([i for i in range(N+1)],ERR_test,'r')
plt.xlabel('after this many trees')
plt.ylabel('ERR')
plt.legend(('train','test'),loc=4)
plt.savefig('./ERR'+str(eta)+'.png')
