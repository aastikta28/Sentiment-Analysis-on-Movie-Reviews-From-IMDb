import input
import svm
import randomForest
import naiveBayes 
import tf_idf
import baggingClassifier
import adaboost
import sklearn

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.feature_extraction import DictVectorizer

import scipy.sparse as ss
import numpy as np

def pca_preprocess(train, matrix):
    n_components = 100
    pca = PCA(n_components)
    pca.fit(train.todense())
    return pca.transform(matrix.todense())

def lca(train, matrix):
    lsa = TruncatedSVD(n_components=100)
    lsa.fit(train)
    return lsa.transform(matrix)

def cross_validation_print(model, train, label, val):
    scores = sklearn.cross_validation.cross_val_score(model, train, label, cv=val)
    print("Scores", str(scores), "mean", str(np.mean(scores)))

def input_preprocess(training_matrix, testing_matrix):
    combine = ss.csc_matrix(np.vstack([training_matrix.todense(),testing_matrix.todense()]))
    zero_cols = np.where(np.array(combine.sum(0)).flatten())[0]
    training_matrix = training_matrix[:,zero_cols]
    testing_matrix = testing_matrix[:,zero_cols]
    combine = combine[:,zero_cols]
    return training_matrix, testing_matrix, combine

lca_flag =0 
pca_flag = 0 
svm_flag = 0 
adaboo_flag = 0 
rf_flag = 0 
nb_flag =0 
bagging_flag = 0

if __name__ == "__main__":
    a = raw_input("Enter the arguments ")
    a = a.split()
    if "lca" in a[0]:
        lca_flag = 1
    elif "pca" in a[0]:
        pca_flag = 1
    if "svm" in a[1]:
        svm_flag = 1
    elif "adaboo" in a[1]:
        adaboo_flag = 1
    elif "rf" in a[1]:
        rf_flag = 1
    elif "nb" in a[1]:
        nb_flag = 1
    elif "bagging" in a[1]:
        bagging_flag = 1
    
    training_matrix = input.input("5k_spring_2016_training_dataset.txt", 15000, 40293)
    testing_matrix = input.input("5k_spring_2016_testing_dataset.txt", 15000, 40293)
    training_label = input.label("5k_spring_2016_label_training.txt", 15000)
    
    training_matrix, testing_matrix, combine = input_preprocess(training_matrix, testing_matrix)
    #Getting tf-idf of the matrix
    tf_idf_combine =  tf_idf.tf_idf(combine, combine)
    tf_idf_training_matrix = tf_idf.tf_idf(combine, training_matrix)
    tf_idf_testing_matrix = tf_idf.tf_idf(combine, testing_matrix)
    
    if (lca_flag):
        print("Doing LCA")
        train = lca(tf_idf_combine, tf_idf_training_matrix)
        test = lca(tf_idf_combine, tf_idf_testing_matrix)
    elif (pca_flag):
        print("Doing PCA")
        train = pca_preprocess(tf_idf_combine, tf_idf_training_matrix)
        test = pca_preprocess(tf_idf_combine, tf_idf_testing_matrix)
    else:
        print("Doing No Reduction")
        train = tf_idf_training_matrix
        test = tf_idf_testing_matrix
    
    if (svm_flag):
        print("Doing SVM")
        ans = svm.support_vector_machine(train, training_label, test, 0.01)
    elif (adaboo_flag):
        print("Doing Adaboost")
        ans = adaboost.adaboo(train, training_label, test)
    elif (rf_flag):
        print("Doing RF")
        ans = randomForest.r_forest(training_matrix, training_label, testing_matrix)
    elif (nb_flag):
        print("Doing Naive Bayes")
        ans = naiveBayes.nb(training_matrix, training_label, testing_matrix)
    elif (bagging_flag):
        print("Doing Bagging")
        ans =  baggingClassifier.bagging(training_matrix, training_label, testing_matrix)
    else:
        print("Invalid argument specified")


