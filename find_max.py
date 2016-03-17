# -*- coding: utf-8 -*-
 
f1 = open("5k_spring_2016_testing_dataset.txt", "r")
maxm = 0
for line in f1:
    line = line.split(' ')
    if int(line[1]) >= maxm:
        maxm = int(line[1])
print "testing file max : ", maxm
f1.close()

f1 = open("5k_spring_2016_training_dataset.txt", "r")
maxm = 0
for line in f1:
    line = line.split(' ')
    #print line
    if int(line[1]) >= maxm:
        maxm = int(line[1])
print "training file max : ", maxm
f1.close()