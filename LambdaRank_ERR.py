import numpy as np
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict
import matplotlib.pyplot as plt

######################################
############ PARAMETERS ##############

eta = 0.1	# learning rate
N = 500	# number of epochs

max_que = 200	# DOWNSCALE: only take this many queries per fold
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

def lin(X,w,b):
	return X@w + b

def ReLU(z):
	return np.maximum(z,[[0]])

def lin_backward_pass(dL_dout,w):
	return dL_dout @ w.T

def lin_param_gradients(dL_dout,input):
	return input.T @ dL_dout

def ReLU_backward_pass(dL_dout,input):
	return dL_dout * np.greater(input,[[0]])

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
W1 = [np.random.randn(max_fea,20) / 1000]	# initialise with v small random weights
b1 = [np.full((20),0.1)]
W2 = [np.random.randn(20,10) / 1000]
b2 = [np.full((10),0.1)]
W3 = [np.random.randn(10,1) / 1000]
b3 = [0]

Z0 = lin(fea[phase],W1[-1],b1[-1])
Z1 = ReLU(Z0)
Z2 = lin(Z1,W2[-1],b2[-1])
Z3 = ReLU(Z2)
train_output = np.squeeze(lin(Z3,W3[-1],b3[-1]))
ERR_train = np.zeros(N+1)
for n in range(N):
	print('epoch',n)
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
		start += ndocs

	dsi_dsj    = np.ones((M[phase],1))
	dsi_dZ3 = lin_backward_pass(dsi_dsj,W3[-1])
	dsi_dZ2 = ReLU_backward_pass(dsi_dZ3,Z2)
	dsi_dZ1 = lin_backward_pass(dsi_dZ2,W2[-1])
	dsi_dZ0 = ReLU_backward_pass(dsi_dZ1,Z0)
	dsi_dW3 = lin_param_gradients(dsi_dsj,Z3)  # 10x1
	dsi_dW2 = lin_param_gradients(dsi_dZ2,Z1)  # 20x10
	dsi_dW1 = lin_param_gradients(dsi_dZ0,fea[phase])  # max_feax20
	W3.append(W3[n] - eta*sum(all_lambdas)*dsi_dW3)	# learning step
	W2.append(W2[n] - eta*sum(all_lambdas)*dsi_dW2)	# learning step
	W1.append(W1[n] - eta*sum(all_lambdas)*dsi_dW1)	# learning step
	b3.append(b3[n] - eta*(all_lambdas@dsi_dsj))	# learning step
	b2.append(b2[n] - eta*(all_lambdas@dsi_dZ2))	# learning step
	b1.append(b1[n] - eta*(all_lambdas@dsi_dZ0))	# learning step
	Z0 = lin(fea[phase],W1[-1],b1[-1])
	Z1 = ReLU(Z0)
	Z2 = lin(Z1,W2[-1],b2[-1])
	Z3 = ReLU(Z2)
	train_output = np.squeeze(lin(Z3,W3[-1],b3[-1]))


model_sco = separate_by_query(train_output,que[phase])
for q in range(Q[phase]):	# get final ERR score
	if max(sco[phase][q]) == 0:
		ERR_train[N] += 1
		continue
	ERR_train[N] += get_ERR(q, model_sco[q], phase, only_total=True)
ERR_train /= Q[phase]


#### VALIDATE ###
#phase='val'
#ERR_vali = np.zeros(N+1)
#for n in range(N+1):
#	test_output = np.matmul(fea[phase],W[n])
#	model_sco = separate_by_query(vali_output,que[phase])
#	for q in range(Q[phase]):	# get ERR scores
#		if not IDCG10[phase][q]:
#			ERR_vali[n] += 1
#			continue
#		ERR_vali[n] += get_ERR(q, model_sco[q], phase, only_total=True)
#	if n==N: break
#ERR_vali /= Q[phase]


### PREDICT ###
phase='tst'
ERR_test = np.zeros(N+1)
for n in range(N+1):
	Z1 = ReLU(lin(fea[phase],W1[n],b1[n]))
	Z2 = ReLU(lin(Z1,W2[n],b2[n]))
	test_output = np.squeeze(lin(Z2,W3[n],b3[n]))
	model_sco = separate_by_query(test_output,que[phase])
	for q in range(Q[phase]):	# get ERR scores
		if max(sco[phase][q]) == 0:
			ERR_test[n] += 1
			continue
		ERR_test[n] += get_ERR(q, model_sco[q], phase, only_total=True)
	if n==N: break
ERR_test /= Q[phase]

### PLOTS ###
plt.plot([i for i in range(N+1)],ERR_train,'b')
plt.plot([i for i in range(N+1)],ERR_test,'r')
plt.xlabel('after this many trees')
plt.ylabel('ERR')
plt.legend(('train','test'),loc=4)
plt.savefig('./RankERR'+str(eta)+'.png')
