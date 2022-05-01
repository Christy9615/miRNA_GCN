# coding=UTF-8
import gc
import numpy as np
import GCN.main as gcn
from metrics import model_evaluate
from MLP import MLP_train
import csv
import os
import numpy.linalg as LA
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# impuation with 0
def constructZeroNet(miRNA_dis_matrix):
    miRNA_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[0], miRNA_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[1], miRNA_dis_matrix.shape[1]), dtype=np.int8))
    mat1 = np.hstack((miRNA_matrix, miRNA_dis_matrix))
    mat2 = np.hstack((miRNA_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


# using similarity matrix for imputations
def constructNet(miRNA_dis_matrix, miRNA_matrix, dis_matrix):
    mat1 = np.hstack((miRNA_matrix, miRNA_dis_matrix))
    mat2 = np.hstack((miRNA_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def constructFeatureMat(miRNA_matrix, dis_matrix):
    miRNA_dis_matrix = np.zeros((miRNA_matrix.shape[0], dis_matrix.shape[1]))
    mat1 = np.hstack((miRNA_matrix, miRNA_dis_matrix))
    mat2 = np.hstack((miRNA_dis_matrix.T, dis_matrix))
    return np.vstack((mat1, mat2))


def get_similarity_matrix(A, miRNA_fun_similarity, dis_sem_sililarity):
    row = A.shape[0]
    column = A.shape[1]
    C = np.asmatrix(A)
    gamd = row / (LA.norm(C, 'fro') ** 2)
    kd = np.mat(np.zeros((row, row)))
    km = np.mat(np.zeros((column, column)))
    D = C * C.T
    for i in range(row):
        for j in range(i, row):
            kd[j, i] = np.exp(-gamd * (D[i, i] + D[j, j] - 2 * D[i, j]))
    kd = kd + kd.T - np.diag(np.diag(kd))
    KD = np.asarray(kd)  # Obtain Gaussian interaction profile kernel similarity for disease SD
    none_zero_position = np.where(miRNA_fun_similarity != 0)
    zero_position = np.where(miRNA_fun_similarity == 0)
    miRNA_similarity = np.zeros(miRNA_fun_similarity.shape)
    #miRNA_similarity[none_zero_position] = (miRNA_fun_similarity[none_zero_position] + KD[none_zero_position]) / 2
    miRNA_similarity[none_zero_position]=miRNA_fun_similarity[none_zero_position]
    miRNA_similarity[zero_position] = KD[zero_position]
    gamam = column / (LA.norm(C, 'fro') ** 2)
    E = C.T * C
    for i in range(column):
        for j in range(i, column):
            km[i, j] = np.exp(-gamam * (E[i, i] + E[j, j] - 2 * E[i, j]))
    km = km + km.T - np.diag(np.diag(km))
    KM = np.asarray(km)
    none_zero_position = np.where(dis_sem_sililarity != 0)
    zero_position = np.where(dis_sem_sililarity == 0)
    dis_similarity = np.zeros(dis_sem_sililarity.shape)
    #dis_similarity[none_zero_position] = (dis_sem_sililarity[none_zero_position] + KM[none_zero_position]) / 2
    dis_similarity[none_zero_position] = dis_sem_sililarity[none_zero_position]
    dis_similarity[zero_position] = KM[
        zero_position]  # Obtain Gaussian interaction profile kernel similarity for miRNA SM
    return miRNA_similarity, dis_similarity


def get_similarity_matrix2(miRNA_fun_matrix, dis_sem_matrix, miRNA_LL_sim, dis_LL_matrix):
    none_zero_position = np.where(miRNA_fun_matrix != 0)
    zero_position = np.where(miRNA_fun_matrix == 0)
    miRNA_similarity = np.zeros(miRNA_fun_matrix.shape)
    miRNA_similarity[none_zero_position] = (miRNA_fun_matrix[none_zero_position] + miRNA_LL_sim[none_zero_position]) / 2
    miRNA_similarity[zero_position] = miRNA_LL_sim[zero_position]
    none_zero_position = np.where(dis_sem_matrix != 0)
    zero_position = np.where(dis_sem_matrix == 0)
    dis_similarity = np.zeros(dis_sem_matrix.shape)
    dis_similarity[none_zero_position] = (dis_sem_matrix[none_zero_position] + dis_LL_matrix[none_zero_position]) / 2
    dis_similarity[zero_position] = dis_LL_matrix[
        zero_position]  # Obtain Gaussian interaction profile kernel similarity for miRNA SM
    return miRNA_similarity, dis_similarity

# construct graphs for miRNA-drug 
def WKNKN( MD_mat, MM_mat, DD_mat, K, r ):

    rows,cols=MD_mat.shape
    y_m=np.zeros((rows,cols))
    y_d=np.zeros((rows,cols))

    knn_network_m = KNN( MM_mat, K ) #for miRNA
    for i in range(0,rows):
            w=np.zeros((1,K))
            # sort_m,idx_m=np.sort(knn_network_m[i,:],2,'descend')
            sort_m=np.sort(knn_network_m[i,:])[::-1]
            idx_m=np.argsort(knn_network_m[i,:])[::-1]
            sum_m=sum(sort_m[0:K])
            for j in range(0,K) :
                w[0,j]=r**(j-1)*sort_m[j]
                y_m[i,:] =  y_m[i,:]+ w[0,j]* MD_mat[idx_m[j],:]
            if sum_m !=0:
                y_m[i,:]=y_m[i,:]/sum_m

    knn_network_d = KNN( DD_mat , K )  #for disease
    for i in range(0,cols):
            w=np.zeros((1,K))
            #[sort_d,idx_d]=np.sort(knn_network_d[i,:],2,'descend')
            sort_d=np.sort(knn_network_d[i,:])[::-1]
            idx_d=np.argsort(knn_network_d[i,:])[::-1]
            sum_d=sum(sort_d[0:K])
            for j in range(0,K):
                w[0,j]=r**(j-1)*sort_d[j]
                y_d[:,i] =  y_d[:,i]+ w[0,j]* MD_mat[:,idx_d[j]].reshape(MD_mat[:,idx_d[j]].shape[0],)
            if sum_d!=0:
                y_d[:,i]=y_d[:,i]/sum_d

    a1=1
    a2=1
    y_md=(y_m*a1+y_d*a2)/(a1+a2)
    MD_mat_new=np.zeros((rows,cols))
    for i in range(0, rows):
         for j in range(0,cols):
             MD_mat_new[i,j]=max(MD_mat[i,j],y_md[i,j])
    return MD_mat_new


def KNN( network , k ):
    rows, cols = network.shape
    network= network-np.diag(np.diag(network))
    knn_network = np.zeros((rows, cols))
    # [sort_network,idx]=np.sort(network,2,'descend')
    sort_network=np.sort(network)[:,::-1]
    idx=np.argsort(network)[:,::-1]
    for i in range(0,rows):
        knn_network[i,idx[i,0:k]]=sort_network[i,0:k]
    return knn_network


'''cross validation'''


def cross_validation_experiment(miRNA_dis_matrix, miRNA_fun_matrix, dis_sem_matrix, seed):
    '''get non zero postions'''
    none_zero_position = np.where(miRNA_dis_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    '''generate random list'''
    np.random.seed(seed)
    positive_randomlist = np.random.permutation(none_zero_row_index.shape[0])

    metric = np.zeros((1, 7))
    k_folds = 5
    print("seed=%d, evaluating miRNA-disease...." % (seed))

    for k in range(k_folds):
        print("------this is %dth cross validation------" % (k + 1))
        if k != k_folds - 1:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds):(k + 1) * int(
                len(none_zero_row_index) / k_folds)]
        else:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds)::]

        '''record row and col of test sets'''
        positive_test_row = none_zero_row_index[positive_test]
        positive_test_col = none_zero_col_index[positive_test]
        train_miRNA_dis_matrix = np.matrix(miRNA_dis_matrix, copy=True)
        train_miRNA_dis_matrix[positive_test_row, positive_test_col] = 0

        '''get the new similarity based graph'''
        miRNA_sim_matrix, dis_sim_matrix = get_similarity_matrix(train_miRNA_dis_matrix, miRNA_fun_matrix,
                                                                 dis_sem_matrix)
        # miRNA_LL_sim = sm.fast_calculate(train_miRNA_dis_matrix, int(1 * (train_miRNA_dis_matrix.shape[0] - 1)))
        # dis_LL_matrix = sm.fast_calculate(train_miRNA_dis_matrix.T,
        #                                   int(1 * (train_miRNA_dis_matrix.shape[1] - 1)))
        # miRNA_sim_matrix2, dis_sim_matrix2 = get_similarity_matrix2(miRNA_fun_matrix, dis_sem_matrix, miRNA_LL_sim,
        #                                                             dis_LL_matrix)

        train_matrix_n = np.matrix(WKNKN(train_miRNA_dis_matrix, miRNA_fun_matrix, dis_sem_matrix, 50, 1))  # 补谱方法
        miRNA_disease_matrix_net = constructZeroNet(train_matrix_n)
        miRNA_disease_emb = gcn.get_gcn_emb(miRNA_disease_matrix_net,constructZeroNet(train_miRNA_dis_matrix))
        miRNA_len = miRNA_dis_matrix.shape[0]

        miRNA_emb_matrix = np.array(miRNA_disease_emb[0:miRNA_len, 0:])
        dis_emb_matrix = np.array(miRNA_disease_emb[miRNA_len::, 0:])
        miRNA_fun_matrix_emb = gcn.get_gcn_emb(np.mat(miRNA_sim_matrix) - np.identity(miRNA_sim_matrix.shape[0])
                                               )
        dis_sim_matrix_emb = gcn.get_gcn_emb(np.mat(dis_sim_matrix) - np.identity(dis_sim_matrix.shape[0]))
        # miRNA_fun_matrix_emb2 = gcn.get_gcn_emb(np.mat(miRNA_LL_sim) - np.identity(miRNA_LL_sim.shape[0]))
        # dis_sim_matrix_emb2 = gcn.get_gcn_emb(np.mat(dis_LL_matrix) - np.identity(dis_LL_matrix.shape[0]))

        '''training the whole matrix'''
        # train_position = np.where(train_miRNA_dis_matrix != 2)
        # train_row = train_position[0]
        # train_col = train_position[1]
        zero_position=np.where(train_miRNA_dis_matrix == 0)
        zero_position_row=zero_position[0]
        zero_position_col=zero_position[1]
        negative_randomlist = np.random.permutation(zero_position_row.shape[0])
        train_none_zero_position=np.where(train_miRNA_dis_matrix == 1)
        train_none_zero_position_row=train_none_zero_position[0]
        train_none_zero_position_col=train_none_zero_position[1]
        train_zero_position_row=zero_position_row[negative_randomlist[0:len(train_none_zero_position_row)]]
        train_zero_position_col=zero_position_col[negative_randomlist[0:len(train_none_zero_position_row)]]
        train_row=np.append(train_none_zero_position_row,train_zero_position_row)
        train_col=np.append(train_none_zero_position_col,train_zero_position_col)

        '''training for features and labels'''
        train_feature_matrix = np.hstack((np.hstack((miRNA_emb_matrix[train_row] ,dis_emb_matrix[train_col])),
                                          np.hstack((miRNA_fun_matrix_emb[train_row] , dis_sim_matrix_emb[train_col]))))
        

        train_label_vector = train_miRNA_dis_matrix[train_row, train_col]
        test_position = np.where(train_miRNA_dis_matrix == 0)
        test_row = test_position[0]
        test_col = test_position[1]

        '''put test data and labels in one matrix'''
        test_feature_matrix = np.hstack((np.hstack((miRNA_emb_matrix[test_row] , dis_emb_matrix[test_col])),
                                         np.hstack((miRNA_fun_matrix_emb[test_row] ,dis_sim_matrix_emb[test_col]))))
     
        test_label_vector = miRNA_dis_matrix[test_row, test_col]

        train_feature_matrix = np.array(train_feature_matrix)
        train_label_vector = np.array(train_label_vector)
        test_feature_matrix = np.array(test_feature_matrix)

        
        predict_y_proba = MLP_train.main(train_feature_matrix, test_feature_matrix, train_label_vector.T)
        this_metric = model_evaluate(test_label_vector, predict_y_proba.reshape(test_label_vector.shape))
        print(this_metric)
        metric += this_metric

        # del clf
        del train_feature_matrix
        del train_label_vector
        del test_feature_matrix
        del test_label_vector
        del this_metric
        gc.collect()

    print(metric / k_folds)

    metric = np.array(metric / k_folds)

    name = 'result/miRNA_disease_seed=' + str(seed) + '.csv'
    np.savetxt(name, metric, delimiter=',')

    return metric

def readxlsx(file_path):
    df = pd.read_excel(
     file_path,
    #  engine='openpyxl',
    header=None
     )
    return df.values

def get_fun_sem_matrix():
    m1 = readxlsx('./data/Disease semantic similarity matrix 1.xlsx')
    m2 = readxlsx('./data/Disease semantic similarity matrix 2.xlsx')
    SS = (m1+m2)/2

    FS = readxlsx('./data/miRNA functional similarity matrix.xlsx')
    return FS,SS


if __name__ == "__main__":
    miRNA_dis_matrix = []
    with open("data/miRNA_disease_matrix.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for i in reader:
            i = [float(x) for x in i]
            miRNA_dis_matrix.append(i)
    csvfile.close()
    miRNA_dis_matrix = np.mat(miRNA_dis_matrix).T

    miRNA_fun_matrix, dis_sem_matrix = get_fun_sem_matrix()

    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 10

    for i in range(circle_time):
        result += cross_validation_experiment(miRNA_dis_matrix, miRNA_fun_matrix, dis_sem_matrix, i)

    average_result = result / circle_time
    print(average_result)
    np.savetxt('result/avg_miRNA_disease.csv', average_result, delimiter=',')
