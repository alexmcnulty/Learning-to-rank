import numpy as np
from sklearn import datasets
import math
import pickle


def loader():
    # train = datasets.load_svmlight_file('../MSLR-WEB10K/Fold1/train.txt', multilabel=False, query_id=True)
    test = datasets.load_svmlight_file('Fold1/test.txt', multilabel=False, query_id=True)
    # valid = datasets.load_svmlight_file('../MSLR-WEB10K/Fold1/vali.txt', multilabel=False, query_id=True)
    return test


def pred_order(o1_test):
    all_pos = o1_test - np.min(o1_test)

    order = np.argsort(all_pos)

    _, first = np.where(order == 4)
    _, second = np.where(order == 3)
    _, third = np.where(order == 2)
    _, fourth = np.where(order == 1)
    _, fith = np.where(order == 0)

    return first, second, third, fourth, fith


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


def all_rank(qd_col, output, true):
    start = 0
    print('b')
    #     first, second, third, fourth, fith = pred_order(output)
    true_order_store = []
    pred_order_store = []
    print('c')
    for i in range(len(qd_col)):
        end = start + qd_col[i]
        true_qid_rel = true[start:end]  ##### this is then the relevance of every doc in that qid
        pred_qid_rel = output[start:end]
        #         sec_rel = second[start:end]
        #         third_rel = third[start:end]
        #         fourth_qid  = fourth[start:end]
        #         fith_qid = fith[start:end]
        a, f = zip(*sorted(zip(pred_qid_rel, true_qid_rel)))

        tru_order = sorted(true_qid_rel)
        true_order_store.append(tru_order)
        pred_order_store.append(f)
        start = end

    return np.asarray(true_order_store), np.asarray(pred_order_store)


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


def total_err(pred_order):
    err_store = []
    for i in range(len(pred_order)):
        documents = pred_order[i]
        err = err_calc(documents)
        err_store.append(err)
    return err_store


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

def rand(test):
    x = 0 * np.ones((len(test[1]), 1))
    y = np.ones((len(test[1]), 1))
    z = 2 * np.ones((len(test[1]), 1))
    i = 3 * np.ones((len(test[1]), 1))
    j = 4 * np.ones((len(test[1]), 1))

    random = np.concatenate((x, y, z, i, j), axis=1)

    rand = np.random.uniform(low=0.0, high=1.0, size=(len(test[1]), 5))
    row_sums = rand.sum(axis=1)
    rand = rand / row_sums[:, np.newaxis]

    score = np.sum((rand * random), axis=1)
    return score



def main():
    test = loader()

    rand_pred = rand(test)

    class_true = test[1]
    qid_list = test[2]

    qd_col = qid_counter(qid_list)  ### this then gives the numbers of a given qid pair

    true_order, pred_ord = all_rank(qd_col, rand_pred, class_true)

    ndcg = total_dcg(true_order, pred_ord)
    print("NDCG@10 mean =", np.mean(ndcg))
    print("NDCG@10 max =", np.max(ndcg))
    print("NDCG@10 min =", np.min(ndcg))
    print("NDCG@10 std =", np.std(ndcg))

    err = total_err(pred_ord)
    print("ERR mean =", np.mean(err))
    print("ERR max =", np.max(err))
    print("ERR min =", np.min(err))
    print("ERR std =", np.std(err))
    print()

    with open('base_metrics.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([ndcg, err], f)

main()
