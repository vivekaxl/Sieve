import pickle, decimal


def euclidean_distance(list1, list2):
    assert(len(list1) == len(list2)), "The points don't have the same dimension"
    distance = sum([(i - j) ** 2 for i, j in zip(list1, list2)]) ** 0.5
    assert(distance >= 0), "Distance can't be less than 0"
    return distance

def get_hotspot_scores(data):
    distance_matrix = [[-1 for _ in xrange(len(data))] for _ in xrange(len(data))]
    for i in xrange(len(data)):
        for j in xrange(len(data)):
            if distance_matrix[i][j] == -1 and i != j:
                distance_matrix[i][j] = euclidean_distance(data[i], data[j])
                distance_matrix[j][i] = distance_matrix[i][j]
            elif distance_matrix[i][j] == -1 and i == j:
                distance_matrix[j][i] = 1
            else:
                pass
    hotspot_scores = [sum(distance_matrix[i]) for i in xrange(len(data))]
    # print "Done calculating hotspot scores"
    return hotspot_scores


clusters = pickle.load( open("grid.p", "rb"))

print "Length of Cluster: ", len(clusters)
import pdb
pdb.set_trace()

for i, cluster in enumerate(clusters):
    t_cluster = [list(c) for c in cluster]
    scores = get_hotspot_scores(t_cluster)
    sorted_index = sorted(range(len(scores)), key=lambda k: scores[k])
    print list(cluster[sorted_index[0]])


