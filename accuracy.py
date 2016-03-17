# -*- coding: utf-8 -*-
import numpy as np

#####################################################################33

act_label = input.label("5k_spring_2016_label_testing.txt", 15000)
lca_adaboo_label = input.label("5k_adaboost.txt", 15000)

print len(act_label)

acc = np.sum(act_label == lca_adaboo_label)
acc = acc + 0.0
print "total correctly identified by adaboost lca : ", acc
print "Accuracy for Adaboost is: ", (acc)/len(lca_adaboo_label)*100
