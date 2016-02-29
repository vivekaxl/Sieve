from __future__ import division
estimators = [50, 70,  90,  110, 130, 150]
max_features = [0.1,  0.4, 0.7,  1.0]
min_samples_split = [1,  9,  17]
min_samples_leaf = [2, 4, 6, 8, 10]
max_leaf_nodes = [10, 15, 20, 25, 30, 35, 40, 45, 50]

count = 1
for a in estimators:
    for b in max_features:
        for c in min_samples_split:
            for d in min_samples_leaf:
                for e in max_leaf_nodes:
                    print a, ",", b, ",", c, ",", d, ",", e
