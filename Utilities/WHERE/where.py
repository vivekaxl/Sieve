from __future__ import division
from os import walk
from pdb import set_trace
from random import randint as randi, seed as rseed

import numpy as np
import pandas as pd

__author__ = 'rkrsn'


def where(data):
    """
    Recursive FASTMAP clustering.
    """
    rseed(0)
    if isinstance(data, pd.core.frame.DataFrame):
        data = data.as_matrix()
    if not isinstance(data, np.ndarray):
        raise TypeError('Incorrect data format. Must be a pandas Data Frame, or a numpy nd-array.')

    N = np.shape(data)[0]
    clusters = []
    norm = 1  # to override the error due to rahuls code. No normalization in place

    def aDist(one, two):
        return np.sqrt(np.sum((np.array(one[:-1]) / norm - np.array(two[:-1]) / norm) ** 2))

    def farthest(one, rest):
        return sorted(rest, key=lambda F: aDist(F, one))[-1]

    def recurse(dataset, step=0):
        print "STEP: ", step, " Length: ", len(dataset)
        R, C = np.shape(dataset)  # No. of Rows and Col
        # Find the two most distance points.
        one = dataset[randi(0, R - 1)]
        pole1 = list(farthest(one, dataset))
        pole2 = list(farthest(pole1, dataset))

        # Project each case on
        def proj(test):
            a = aDist(pole1, test)
            b = aDist(pole2, test)
            c = aDist(pole1, pole2)
            x = (a ** 2 - b ** 2 + c ** 2) / (2 * c)
            y = (a ** 2 - x ** 2) ** 0.5
            return [x, y]

        def generate_cluster(dataset):
            x_y_scores = map(lambda x: proj(x), dataset)
            x_med = sorted(x_y_scores, key=lambda x:x[0])[int(len(x_y_scores)/2)][0]
            y_med = sorted(x_y_scores, key=lambda x:x[1])[int(len(x_y_scores)/2)][1]

            def define_cluster(test):
                if test[0] <= x_med:
                    if test[1] <= y_med: return 1
                    else: return 2
                else:
                    if test[1] <= y_med: return 2
                    else: return 3

            no_clusters = map(lambda x: define_cluster(x), x_y_scores)
            return no_clusters

        if R < np.sqrt(N) or step == 3:
            clusters.append(dataset)
        else:
            clusters_numbers = generate_cluster(dataset)
            seperated_clusters = [[] for _ in xrange(4)]

            for d, index in zip(dataset, clusters_numbers): seperated_clusters[index].append(d)
            for index in xrange(4):
                if len(seperated_clusters[index]) == 0: continue
                _ = recurse(seperated_clusters[index], step+1)

    recurse(data)
    return clusters


def _test(dir='../Data/'):
    files = []
    for (dirpath, _, filename) in walk(dir):
        for f in filename:
            print dirpath + f
            df = pd.read_csv(dirpath + f)
            headers = [h for h in df.columns if '$<' not in h]
            files.append(df[headers])


    "For N files in a project, use 1 to N-1 as train."
    train = pd.concat(files)
    clusters = where(train)
    from pickle import dump, load
    dump( clusters, open( "grid.p", "wb" ) )

    # ----- ::DEBUG:: -----
    set_trace()

if __name__ == '__main__':
    _test()
