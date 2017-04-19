import numpy as np
from sklearn import datasets
from scipy.misc import logsumexp
import math
import pickle

######## ---------- LOADING THE DATA ---------- ########
def loader():
    train = datasets.load_svmlight_file('Fold1/train.txt', multilabel=False, query_id=True)
    test = datasets.load_svmlight_file('Fold1/test.txt', multilabel=False, query_id=True)
    valid = datasets.load_svmlight_file('Fold1/vali.txt', multilabel=False, query_id=True)
    return train, test,valid

######## ---------- LOGISTIC REGRESSION ---------- ########
class linear(object):
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def fwd_pass(self, x):
        return np.matmul(x,self.weights) + self.bias

    def bwd_pass(self, dl_dy):
        dl_dx = np.matmul(dl_dy, self.weights.T)
        return dl_dx

    def param_grad(self, dl_dy, x, batch_size):
        dl_dw = np.matmul(np.reshape(x, [batch_size, -1,1]), np.reshape(dl_dy, [batch_size,1,-1]))
        dl_db = dl_dy
        return dl_dw, dl_db


class soft_xent(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fwd_pass(self):
        soft_out = np.exp(self.x - logsumexp(self.x, axis=1, keepdims=True))
        return soft_out

    def bwd_pass(self):
        dl_dx = self.fwd_pass() - self.y
        return dl_dx


def predictor(x):
    p1 = np.exp(x - logsumexp(x, axis=1, keepdims=True))# this is the softmax output
    return np.argmax(p1,axis=1)


def acc_calc(x, target):
    counter = 0
    pred = predictor(x)
    target = np.argmax(target, axis =1)
    N = len(target)
    for i in range(N):
        if pred[i] == target [i]:
            counter = counter + 1
    return counter/N


def one_hotter(data):
    y = data.reshape(-1,)
    numbers = [int(x) for x in y]
    b = np.zeros((len(y), int(y.max()+1)))
    x = np.arange(int(len(y)))
    b[x, numbers] = 1
    return b

######## ---------- RETURNS NUMBER PER QID ---------- ########
def qid_counter(qid):
    qid_number = []
    c=1
    for i in range(len(qid)):
        if i == len(qid)-1:
            qid_number.append(c)
        elif qid[i] == qid[i+1]:
            c+=1
        else:
            qid_number.append(c)
            c = 1
    return np.asarray(qid_number)



######## ---------- TRAINING ---------- ########
def optimiser(num_epochs, learning_rate, lamb, data):
    
    x_train = data[0][0]
    y_train = one_hotter(data[0][1])
    
    x_test = data[1][0]
    y_test = one_hotter(data[1][1])
    
    qid_train = data[0][2]
    qid_test = data[1][2]
    
    #this returns the number of documents per query
    qd_col_train = qid_counter(qid_train)
    
    num_features = x_train.shape[1]
    num_labels = 5
    
    #---initial weights---#
    w1 = np.random.randn(num_features,num_labels)
    b1 = np.random.randn(num_labels)

    best = 0
    for epoch in range(num_epochs):
        start=0
        for mini_batch in range(len(qd_col_train)):
            #this adjusts it so that each batch is for a given query
            end = start + qd_col_train[mini_batch] 
            # batch_size = end - start
            
            x_batch = x_train[start:end]
            y_batch = y_train[start:end]
            x_batch = x_batch.toarray()
            
            batch_size = end - start

            ###### FORWARD PASS ######
            #---Linear Layer---#
            linear_layer = linear(w1, b1)
            linear_output = linear_layer.fwd_pass(x_batch)
            
            #---Softmax Xent layer---#
            xent = soft_xent(linear_output,y_batch)
            
            ###### BACKWARD PASS ######
            gradient = xent.bwd_pass()
            
            #---get param gradient---#
            dl_dw, dl_db = linear_layer.param_grad(gradient, x_batch, batch_size)
            
            #---update params---#
            w1 = w1 - learning_rate * ((np.sum(dl_dw,0)+lamb*w1)/batch_size)
            b1 = b1 - learning_rate * (np.sum(dl_db, 0)/batch_size)
            
            start = end
            
        if epoch%5==0:
            l1_final = linear(w1,b1)
            o1_test = l1_final.fwd_pass(x_test.toarray())
            test_acc = acc_calc(o1_test, y_test)
            if test_acc>=best:
                best=test_acc
                w_best = w1
                b_best = b1
    print(best)
    return w_best, b_best


######## ---------- GET THE PREDICTION ORDER ---------- ########
def pred_order(o1_test):
    all_pos = o1_test - np.min(o1_test)

    order = np.argsort(all_pos)

    _, first = np.where(order == 4)
    _, second = np.where(order == 3)
    _, third = np.where(order == 2)
    _, fourth = np.where(order == 1)
    _, fith = np.where(order == 0)

    return first, second, third, fourth, fith


######## ---------- RETURNS THE RANKING FROM THE PREDICTION ---------- ########
def all_rank(qd_col,output,true ):
    start = 0
    print('b')
    first, second, third, fourth, fith = pred_order(output)
    true_order_store = []
    pred_order_store = []
    print('c')
    for i in range(len(qd_col)):
        end = start + qd_col[i]
        true_qid_rel = true[start:end] ##### this is then the relevance of every doc in that qid
        pred_qid_rel = first[start:end]
        sec_rel = second[start:end]
        third_rel = third[start:end]
        fourth_qid  = fourth[start:end]
        fith_qid = fith[start:end]
        a, b, c, d,e,f  = zip(*sorted(zip(pred_qid_rel, sec_rel, third_rel,fourth_qid, fith_qid,true_qid_rel)))

        tru_order = sorted(true_qid_rel)
        true_order_store.append(tru_order)
        pred_order_store.append(f)
        start = end
        
    return np.asarray(true_order_store), np.asarray(pred_order_store) 


######## ---------- METRICS ---------- ########
def dcg_calc(rank):
    rank = rank[-10:]
    rank = rank[::-1]
#     print(rank)
    dcg = 0
    for i in range(len(rank)):
        rel = rank[i]
        j=i+1
        dcg += (2**rel -1)/ np.log2(1+j)
    return dcg


def total_dcg(truth, pred):
    ndcg = []
    for i in range(len(truth)):
        true_dcg = dcg_calc(truth[i])
        pred_dcg = dcg_calc(pred[i])
        y = (pred_dcg/true_dcg)
        x=float((pred_dcg/true_dcg))
        if math.isnan(x):
            y = 0
        ndcg.append(y)
    return ndcg


def err_calc(documents):
    p = 1
    ERR = 0
    documents = documents[::-1]
    for i in range(len(documents)):
        j = i+1
        rel = documents[i]
        R = (2**rel -1)/(2**4)
        ERR += p*(R/j)
        p = p*(1-R)
    return ERR


def total_err(pred_order):
    err_store = []
    for i in range(len(pred_order)):
        documents = pred_order[i]
        err = err_calc(documents)
        err_store.append(err)
    return err_store




######## ---------- MAIN TO PUT IT ALL TOGETHER ---------- ########
def main():
    train, test, valid = loader()
    data = [train,test,valid]
    weights, bias = optimiser(100,0.1, 1, data)

    #### this gets the ranking predictions ##########
    l1_final = linear(weights,bias)
    o1_test = l1_final.fwd_pass(data[1][0].toarray())


    #### these are the values we need to obtain the various metric ###
    class_true = test[1]
    qid_list = test[2]
    
    qd_col = qid_counter(qid_list) ### this then gives the numbers of a given qid pair
    
    true_order, pred_ord = all_rank(qd_col,o1_test,class_true)

    ndcg = total_dcg(true_order, pred_ord)
    print("NDCG@10 mean =", np.mean(ndcg))
    print("NDCG@10 max =",np.max(ndcg))
    print("NDCG@10 min =",np.min(ndcg))
    print("NDCG@10 std =",np.std(ndcg))
    
    err = total_err(pred_ord)
    print("ERR mean =", np.mean(err))
    print("ERR max =",np.max(err))
    print("ERR min =",np.min(err))
    print("ERR std =",np.std(err))
    print()
    print()

    with open('logreg_metrics.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ndcg, err], f)
    
    return ndcg, err



main()



# optimum values are lr = 0.1, lamb =1
######## ---------- NOTE NEED TO CHANGE OPTIMISER TO TEST ON VALID IF DOING GRID SEARCH ---------- ########
def grid_search():
    train, test, valid = loader()
    data = [train,test,valid]
    # lamb_list = [0, 0.1, 1, 10]
    # lr_list = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    lamb_list = [1, 5, 10]
    lr_list = [0.1, 0.05, 0.01]
    for lamb in lamb_list:
        for lr in lr_list:
            score_store = []
            epoch_store = []
            for i in range(3):
                best, i = optimiser(100, lr,lamb, data)
                print(best)
                print(i)
                score_store.append(best)
                epoch_store.append(i)
            print("best for lr =", lr, " lambda = ", lamb, ' is ')
            print("max was " , np.max(score_store), " ")
            print("mean was " , np.mean(score_store), " ")
            print(score_store)
            print()
            print(epoch_store)
            print()

# grid_search()



