from __future__ import division
estimators = xrange(50, 151, 5)
max_features = range(1, 10, 1)
min_samples_split = xrange(1,21, 2)
min_samples_leaf = xrange(2,21, 2)
max_leaf_nodes = xrange(10,51, 5)

count = 1
for a in estimators:
    for b in max_features:
        for c in min_samples_split:
            for d in min_samples_leaf:
                for e in max_leaf_nodes:
                    print a, ",", b/10, ",", c, ",", d, ",", e
