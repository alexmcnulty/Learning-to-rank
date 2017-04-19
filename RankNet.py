
# coding: utf-8

# In[1]:

import sklearn
import numpy as np
import collections
import math
import random
import time
import tensorflow as tf
import os
import copy
import itertools
import pdb
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_file
import warnings
import matplotlib.pyplot as plt
import matplotlib
import os
warnings.filterwarnings('ignore')


# In[2]:

# 2000 queries per each data subset. Use less for intitial training
max_que =2000	# DOWNSCALE: only take this many queries per fold
max_fea = 136	# DOWNSCALE: only use this many features


# In[3]:

mem = Memory("./mycache")	# cache to binary for faster load
@mem.cache
def get_data(file):
    return load_svmlight_file(file,query_id=True)

def extract_data(phase,folds):	# put into useful structures
    fea = np.empty((0,max_fea))
    sco = []
    feat_by_query = []
    q_id = []
    que = np.empty((0,),dtype=np.int32)
    M = 0
    relevance=[]   
    for i in folds:
        features,scores,queries = S[i]
        relevance.extend(scores)
        scores = [np.array(x,dtype=np.int32) for x in separate_by_query(scores,queries)[:max_que]]
        stop = sum(len(query) for query in scores)	# number of rows for max_que
        q_id.append(stop)        
        M += stop
        fea = np.concatenate((fea,features[:stop,:max_fea].toarray()))
        sco.extend(scores)
        
        
        que = np.concatenate((que,queries[:stop]))
    return fea,sco,que,M,len(sco),q_id, np.asarray(relevance)

def separate_by_query(scores,queries):	# from 2 lists to 1 list of lists
    scoreslist = []
    total_scores = []
    current_query = None
    for score, query in zip(scores, queries):
        if query != current_query:	# works since query info is contiguous
            scoreslist.append([])
            current_query = query
        scoreslist[-1].append(score)
    return [np.array(x,dtype=np.float32) for x in scoreslist]

def feat_separate_by_query(scores,queries):	# from 2 lists to 1 list of lists
    scoreslist = []
    total_scores = []
    current_query = None
    for score, query in zip(scores, queries):
        if query != current_query:	# works since query info is contiguous
            scoreslist.append([])
            current_query = query
        scoreslist[-1].append(score)
    return [np.array(x,dtype=np.float32) for x in scoreslist]


# In[4]:

S = [None]	# dummy so index isn't confusing
for i in range(5):
    S.append(get_data('./MSLR-WEB10K/S'+str(i+1)+'.txt'))
train_fea,train_sco,train_que,M_train,Q_train,Q_id_train,relevance_train = extract_data('train',[1,2,3,4])
#vali_fea, vali_sco, vali_que, M_vali, Q_vali,Q_id_vali, relevance_valid= extract_data('vali',[3])
test_fea, test_sco, test_que, M_test, Q_test, Q_id_test,relevance_test  = extract_data('test',[5])
print('data loaded')


# In[5]:

len(train_sco)


# In[6]:

# need placeholders for the inputs to train, x, and the true labels
max_fea = 136
x  = tf.placeholder( tf.float32, shape =[None, max_fea],name ='x_labels')
# Create a placeholder for the y labels
relevance_scores= tf.placeholder( tf.float32, shape = [None, 1],name = 'y_labels')


# In[7]:

# Need to save the model, weights and biases varibles
name = 'test'
# Suggested Directory to use
save_MDir = 'models/'


#create the directory if it does not exist already
if not os.path.exists(save_MDir):
    os.makedirs(save_MDir)

save_model = os.path.join(save_MDir,'best_accuracy_'+name)    


# In[8]:

def optimize_cost(x, relevance_labels, learning_rate, n_hidden, n_layers):

    n_data = tf.shape(x)[0]

    def get_variables():
        variables = [tf.Variable(tf.random_normal([max_fea, n_hidden], stddev=math.sqrt(2 / (max_fea)))),
            tf.Variable(tf.zeros([n_hidden]))]
        
        if n_layers > 1:
            for i in range(n_layers-1):
                variables.append(tf.Variable(tf.random_normal([n_hidden, n_hidden], stddev=math.sqrt(2 / (n_hidden)))))
                variables.append(tf.Variable(tf.zeros([n_hidden])))
                
        variables.append(tf.Variable(tf.random_normal([n_hidden, 1], stddev=math.sqrt(2 / (n_hidden)))))
        variables.append(tf.Variable(0, dtype=tf.float32))
        return variables

    def score(x, *vars):
        z = tf.contrib.layers.batch_norm(tf.matmul(x, vars[0]) + vars[1])
        if n_layers > 1:
            for i in range(0,n_layers-1):
                z = tf.contrib.layers.batch_norm(tf.matmul(tf.nn.relu(z), vars[2*(i+1)]) + vars[2*(i+1)+1])
        return tf.matmul(tf.nn.relu(z), vars[-2]) + vars[-1]

    vars = get_variables()
    o_ij = score(x, *vars)
    S_ij = tf.maximum(tf.minimum(1., relevance_labels - tf.transpose(relevance_labels)), -1.)
    
    targets = (1 / 2) * (1 + S_ij)
    
    pairwise_o_ij = o_ij - tf.transpose(o_ij)
    cost = tf.nn.sigmoid_cross_entropy_with_logits(pairwise_o_ij, targets)
    cost = tf.reduce_mean((tf.ones([n_data, n_data]) - tf.diag(tf.ones([n_data]))) * cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    def get_score(sess, feed_dict):
        return sess.run(o_ij, feed_dict=feed_dict)

    def run_optimizer(sess, feed_dict):
        sess.run(optimizer, feed_dict=feed_dict)

    return cost, run_optimizer, get_score


# In[9]:

def get_NDCG(ranks,scores,level = 10):
    "Get NDCG @10. Takes in outputs from all docs after being passed through net and takes top 10,"
    " gets there relevance scores and calcuates the NDCG"
    # Takes in outputs from
    top_scores = []
    top_true_scores = []
    # Get top ten ouptputs indexes
    for i in range(len(ranks)):
        top_scores.append(list(reversed(np.argsort(ranks[i])[-10:].tolist())))
        top_true_scores.append(list(reversed(np.argsort(scores[i])[-10:].tolist())))
    top_rels = []
    top_true_rels = []
    
    # get the relevance scores of the top 10 predicted and top 10 actual for the ideal dcg
    for i in range(len(scores)):
        
        top_rels.append(list(scores[i][top_scores[i]])) 
        top_true_rels.append(list(scores[i][top_true_scores[i]])) 
    ndcg_q = []
    
    # For each query loop over the top 10 documents and calculate the dcg and ideal dcg
    for i in range(len(top_rels)):
        current_dcg = 0
        current_ideal = 0
        
        # There exists cases with less than 10 dcouments are associated with the query
        if len(top_rels[i])<10: 
            level = len(top_rels[i])
        for j in range(level):
            current_dcg += ((2**top_rels[i][j]) - 1)/np.log(j+1+1)
            current_ideal+= ((2**top_true_rels[i][j]) - 1)/np.log(j+1+1)
            
        # calculate the ndcg
        if  current_ideal != 0:
            ndcg_q.append(current_dcg/current_ideal)
        else: ndcg_q.append(0)
        level = 10
    
    return ndcg_q


# In[10]:

def get_ERR(ranks,scores,level = 10):
    "Get the Expected reciprical Rank"
    top_scores = []
    
    # Get the top 10 ranked score indexes
    for i in range(len(ranks)):
        top_scores.append(list(reversed(np.argsort(ranks[i])[-10:].tolist())))
        
    top_rels = []
    # get the relvance scores
    for i in range(len(scores)):
        top_rels.append(list(scores[i][top_scores[i]]))
        
    ERR_q = []
    for i in range(len(top_rels)):
        current_err = 0
        if len(top_rels[i])<10: 
            level = len(top_rels[i])
 
        prod = 0
        count = 1
        for r in range(level):
            R = 2**(top_rels[i][r])-1
            prod = (1/count) *(R/2**4)
            for j in range (r):
                Rj = (2**(top_rels[i][j])-1)/16
                prod *= 1-Rj
            current_err +=prod
            count+=1
        ERR_q.append(current_err)
        level = 10    
    return ERR_q


# In[16]:

def get_random_NDCG(ranks,scores,level = 10):
    random_scores = []
    top_true_scores = []
    for i in range(len(ranks)):
        top_true_scores.append(list(reversed(np.argsort(scores[i])[-10:].tolist())))
    top_rels = []
    top_true_rels = []
    #print(len(top_scores))
    for i in range(len(scores)):
        if len(ranks[i])<10:
                top_rels.append(list(ranks[i][:len(ranks[i])]))
        else:
            top_rels.append(list(ranks[i][:10]))
        top_true_rels.append(list(scores[i][top_true_scores[i]])) 
    ndcg_q = []
    #pdb.set_trace()
    for i in range(len(top_rels)):
        current_dcg = 0
        current_ideal = 0
        #print('Running query:{}'.format(i+1))
        if len(top_rels[i])<10: 
            level = len(top_rels[i])
        #print(level)   
        for j in range(level):
            current_dcg += ((2**top_rels[i][j]) - 1)/np.log(j+1+1)
            current_ideal+= ((2**top_true_rels[i][j]) - 1)/np.log(j+1+1)
            #print(top_rels[i][j])
            #print(current_dcg)
        if  current_ideal != 0:
            ndcg_q.append(current_dcg/current_ideal)
        else: ndcg_q.append(0)
        level = 10
    
    return ndcg_q


# In[11]:

def get_test(this_sco,this_feature,i):
    batch_size = 64
    test_ranks = []
    ind = 0
    count_docs = 0
    docs_per_query = len(this_sco[i])

    n_batches = math.floor(docs_per_query/batch_size)
    for batch in range(n_batches):
        #print(j)
        docs = this_feature[ind:ind+batch_size]
        rnks = relevance_train[ind:ind+batch_size]
        # index for all scores
        ind += batch_size
        # index for number of docs passed
        count_docs += batch_size
        feed_dict = {x: np.array(docs, ndmin=2),
                    relevance_scores: np.array(rnks, ndmin=2).T,
                       }
        s = score(sess, feed_dict)

        # get the outputs without optimzing
        #out = sess.run(output, feed_dict)
        s =list(itertools.chain.from_iterable(s))
        #pdb.set_trace()
        test_ranks.append(s)



        # If the remaining batch is less than the normal batchsize
        if batch+1 == n_batches:
            if docs_per_query > count_docs:
                batch_size = (docs_per_query - ind)
                docs = this_feature[ind:ind+batch_size]
                rnks = relevance_train[ind:ind+batch_size]
                ind += batch_size
                feed_dict = {x: np.array(docs, ndmin=2),
                    relevance_scores: np.array(rnks, ndmin=2).T,
                       }   

                s = score(sess, feed_dict)

                # get the outputs without optimzing
                #out = sess.run(output, feed_dict)
                s =list(itertools.chain.from_iterable(s))
                test_ranks.append(s)


    # case where batch size is too big    
    if n_batches == 0:
        batch_size = docs_per_query
        docs = this_feature[ind:ind+batch_size]
        rnks = relevance_train[ind:ind+batch_size]
        ind += batch_size
        feed_dict = {x: np.array(docs, ndmin=2),
            relevance_scores: np.array(rnks, ndmin=2).T,
               }   

        s = score(sess, feed_dict)

        s =list(itertools.chain.from_iterable(s))
        test_ranks.append(s)
    return list(itertools.chain.from_iterable(test_ranks))


# In[18]:

def calc_metrics(total_ranks,sco):
    ndcg_list = get_NDCG(total_ranks,sco)
    ndcg = np.mean(ndcg_list)
    err_list = get_ERR(total_ranks,sco)
    err = np.mean(err_list)
    return ndcg,err,ndcg_list,err_list


# In[13]:

all_features = [train_fea,test_fea]
all_sco = [train_sco, test_sco] 
total_ranks= []
epoch=0


# In[31]:

# taking pairs from the same query
learning_rate = 0.0001
n_hidden = 20
n_layers = 3

cost, optimizer, score = optimize_cost(x, relevance_scores, learning_rate, n_hidden, n_layers)
start = time.time()
saver = tf.train.Saver()

NDCG_epochs,ERR_epochs =[],[]
all_features = [train_fea,test_fea]
all_sco = [train_sco, test_sco] 
n_epochs = 50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for epoch in range(n_epochs):
        
        total_ranks= []
        query_c = 0
        batch_size = 64
        this_feature = all_features[0]
        this_sco = all_sco[0]
        for i in range(len(this_sco)):
            indices = np.random.randint(query_c, query_c + len(this_sco[i]), batch_size)
            query_c+= len(this_sco[i])
            if len(indices) > batch_size:
                indices = indices[:batch_size]

            optimizer(sess, {
                            x: np.array(train_fea[indices], ndmin=2),
                            relevance_scores: np.array(relevance_train[indices], ndmin=2).T,})
            if i%1000 ==0:
                print('Running query {}, total time elapsed: {}'.format(i+1, time.time() - start))
                c = sess.run(cost,{x: np.array(train_fea[indices], ndmin=2),
                                    relevance_scores: np.array(relevance_train[indices], ndmin=2).T,})
                #print(c)
        this_feature = all_features[1]
        this_sco = all_sco[1]   
        ranks = []
        batch_size = 64
        ind = 0
        for i in range(len(this_sco)):
            
            ranks =get_test(this_sco,this_feature,i)

            #append to list for ranks of query
            total_ranks.append(ranks)
            # Each element is a list of ranks for each phase   
        print('caluclating NDCG for epoch: {}'.format(epoch+1))
        ndcg , err,ndcg_RN,err_RN = calc_metrics(total_ranks,all_sco[1])
        print('NDCG:= {}, ERR:= {}\n'.format(ndcg,err))
        #print('ERR vals at this epoch = {}\n'.format(err))
        NDCG_epochs.append(ndcg)
        ERR_epochs.append(err) 
        print('saving_model..')
        #saver.save(sess= sess, save_path = save_model)


# In[63]:

list_nds = []
for i in range(50):
    random_rank = copy.deepcopy(all_sco[1])
    for i in range(len(random_rank)):
        r = random.random()
        random.shuffle(random_rank[i], lambda:r)
    #nd, top_rels = get_random_NDCG(random_rank,all_sco[ndcg])
    nd_list = (get_random_NDCG(random_rank,all_sco[1]))
    nd = np.mean(nd_list)
    list_nds.append(nd_list)
    ERR = np.mean(np.nan_to_num(get_ERR(random_rank,all_sco[1])))
    #ndcg_list.append(nd)
    #err_list.append(ERR)
print('NDCG vals fro random top 10 = {}'.format(nd))
#print('ERR vals at this epoch = {}, {}, {}\n'.format(err_list[0],err_list[1],err_list[2]))


# In[67]:

nd_arr = np.asarray(list_nds)


# In[76]:

nd_rand = np.mean(nd_arr,0).tolist()


# In[97]:

# filename1 = './results/RN_ndcg_'+name+'.npy'
# filename2 = './results/RN_err_'+name+'_.npy'
# filename3 = './results/RN_ndcg_list_'+name+'.npy'
# filename4 = './results/RN_err_list_'+name+'.npy'

# np.save(filename3,(ndcg_RN) )
# np.save(filename4,(err_RN) )
# #
# #np.save(filename3,np.asarray(get_NDCG(all_ranks[ndcg],all_sco[ndcg])) )
# #np.save(filename4,np.asarray(np.nan_to_num(get_ERR(all_ranks[ndcg],all_sco[ndcg]))))


# ## Plot the metric values

# In[86]:

get_ipython().magic('pylab inline')
pylab.rcParams['figure.figsize'] = (10, 6)


# In[ ]:

n_epochs = 50
iterations = n_epochs*len(all_sco[0])
t = np.linspace(0,n_epochs-1,n_epochs)
fig = plt.figure()

plt.plot(t,NDCG_epochs,'b')# plotting t,a separately 

fig.suptitle('Plot of NDCG@10 on test set over 50', fontsize=20)
legend = plt.legend(loc='lower right', shadow=True)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('NDCG Value', fontsize=16)
plt.show()
#fig.savefig('./results/ndcg_50.jpg')


# In[ ]:

n_epochs = 50
iterations = n_epochs*len(all_sco[0])
t = np.linspace(0,n_epochs-1,n_epochs)
fig = plt.figure()

plt.plot(t,ERR_epochs,'b')# plotting t,a separately 

fig.suptitle('Plot of ERR@10 on test set over 50', fontsize=20)
legend = plt.legend(loc='lower right', shadow=True)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('ERR Value', fontsize=16)
plt.show()
#fig.savefig('./results/err_50.jpg')


# ## Test for significane
# Use a t-test and wilcoxons to test if there is a significant difference between the results.
# Calculate a random mean NDCG value for every epoch and take a t test between the random and the actual.

# In[24]:

from scipy import stats


# In[77]:

t,p =stats.ttest_rel(nd_rand,ndcg_RN)


# In[78]:

p


# In[79]:

t_w,p_w = stats.wilcoxon(nd_list, ndcg_RN, zero_method='wilcox', correction=False)


# In[80]:

p_w


# # Load Model

# In[14]:

# Need to save the model, weights and biases varibles
name = 'set5_diff_layers'
# Suggested Directory to use
save_MDir = 'models/'
save_model = os.path.join(save_MDir,'best_accuracy_'+name)    


# In[ ]:

learning_rate = 0.001
n_hidden = 20
n_layers = 2
tf.reset_default_graph()
init = tf.global_variables_initializer()
# need placeholders for the inputs to train, x, and the true labels
max_fea = 136
x  = tf.placeholder( tf.float32, shape =[None, max_fea],name ='x_labels')
# Create a placeholder for the y labels
relevance_scores= tf.placeholder( tf.float32, shape = [None, 1],name = 'y_labels')

cost, optimizer, score = optimize_cost(x, relevance_scores, learning_rate, n_hidden, n_layers)
saver2restore = tf.train.Saver()

saver = tf.train.Saver()

start = time.time()
NDCG_epochs,ERR_epochs =[],[]
all_features = [train_fea,test_fea]
all_sco = [train_sco, test_sco] 
n_epochs = 10
epoch = 0
with tf.Session() as sess:
    sess.run(init)
    saver2restore.restore(sess = sess, save_path= save_model)
    this_feature = all_features[1]
    this_sco = all_sco[1]   
    total_ranks = []
    batch_size = 64
    ind = 0
    for i in range(len(this_sco)):
        ranks = []
        ranks =get_test(this_sco,this_feature,i)

        #append to list for ranks of query
        total_ranks.append(ranks)
        # Each element is a list of ranks for each phase   
    #pdb.set_trace()  
    print('caluclating NDCG for epoch: {}'.format(epoch+1))
    ndcg , err = calc_metrics(total_ranks,all_sco[1])
    print('NDCG:= {}, ERR:= {}\n'.format(ndcg,err))
    #print('ERR vals at this epoch = {}\n'.format(err))
    NDCG_epochs.append(ndcg)
    ERR_epochs.append(err) 
    epoch+=1


# In[ ]:




# In[ ]:



