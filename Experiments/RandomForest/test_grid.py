from __future__ import division
import itertools


class dataholder:
    newid = itertools.count().next

    def __init__(self, decisions, objective):
        self.id = dataholder.newid()
        self.decisions = decisions
        self.objective = objective


def get_data(file):
    contents = []
    lines = open(file, "r").readlines()
    for lineno, line in enumerate(lines):
        if lineno == 0: continue
        content = line.strip().split(",")[3:]
        contents.append(dataholder(content[:-1], 1 if int(content[-1]) > 0 else 0))
    return contents


def get_configurations(file):
    lines = open(file, "r").readlines()
    content = []
    for line in lines: content.append(line.strip().split(","))
    return content


def evaluate_experiment(indep_training, dep_training, indep_testing, dep_testing, configuration):
    def get_F(actual, predicted):
        A = 0
        B = 0
        C = 0
        D = 0

        for a, p in zip(actual, predicted):
            if a == 1:
                if a == p: D += 1
                else: B += 1
            else:
                if p == 1: C += 1
                else: A += 1
        pd = D / (B + D + .001)
        prec = D / (C+ D + .001)
        return 2 * (pd * prec) / (pd + prec + .001)

    # Import the random forest package
    from sklearn.ensemble import RandomForestClassifier

    # Create the random forest object which will include all the parameters
    # for the fit
    forest = RandomForestClassifier(n_estimators = int(configuration[0]),
                                    max_features = configuration[1],
                                    min_samples_split = configuration[2],
                                    min_samples_leaf = configuration[3],
                                    max_leaf_nodes = int(configuration[4]),
                                    random_state=1
                                    )
    # import pdb
    # pdb.set_trace()

    # Fit the training data to the Survived labels and create the decision trees
    forest = forest.fit(indep_training, dep_training)

    # Take the same decision trees and run it on the test data
    output = forest.predict(indep_testing)

    return get_F(dep_testing, output)


def criteria(name, training=[], testing=[], configuration_file=None):
    indep_training_data = []
    dep_training_data = []

    indep_tuning_data = []
    dep_tuning_data = []

    indep_testing_data = []
    dep_testing_data = []

    for train in training:
        contents = get_data(train)
        indexes = [i for i in xrange(len(contents))]
        from random import shuffle
        shuffle(indexes)
        train_data = [contents[i] for i in indexes[:int(len(indexes)/2)]]
        tune_data = [contents[i] for i in indexes[int(len(indexes)/2):]]
        indep_training_data.extend([c.decisions for c in train_data])
        dep_training_data.extend([c.objective for c in train_data])

        indep_tuning_data.extend([c.decisions for c in tune_data])
        dep_tuning_data.extend([c.objective for c in tune_data])

    for test in testing:
        contents = get_data(test)
        indep_testing_data.extend([c.decisions for c in contents])
        dep_testing_data.extend([c.objective for c in contents])

    configurations = [map(float,c) for c in get_configurations(configuration_file)]

    scores = [evaluate_experiment(indep_training_data, dep_training_data, indep_tuning_data, dep_tuning_data, configuration)
              for configuration in configurations]

    sorted_scores_idx = sorted(range(len(scores)), key=lambda k: scores[k])

    testing_configuration = configurations[sorted_scores_idx[-1]]

    testing_score = evaluate_experiment(indep_training_data, dep_training_data, indep_testing_data, dep_testing_data, testing_configuration)

    print "Number of Evaluations: ", len(configurations)
    print name, "|Best Possible Score of : ",  round(testing_score, 3), \
        " can be achieved using: ", testing_configuration



if __name__ == "__main__":

    print "ANT"
    criteria(name="1", training=["./Data/ant/ant-1.3.csv"],
             testing=["./Data/ant/ant-1.5.csv"],
             configuration_file="./configuration_grid.txt")
    # criteria(name="2", training=["./Data/ant/ant-1.4.csv"], tuning=["./Data/ant/ant-1.5.csv"],
    #          testing=["./Data/ant/ant-1.6.csv"],
    #          configuration_file="./configurations.txt")
    # criteria(name="3", training=["./Data/ant/ant-1.5.csv"], tuning=["./Data/ant/ant-1.6.csv"],
    #          testing=["./Data/ant/ant-1.7.csv"],
    #          configuration_file="./configurations.txt")
    #
    # print "CAMEL"
    # criteria(name="1", training=["./Data/camel/camel-1.0.csv"], tuning=[ "./Data/camel/camel-1.2.csv"],
    #      testing=["./Data/camel/camel-1.4.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/camel/camel-1.2.csv"], tuning=[ "./Data/camel/camel-1.4.csv"],
    #      testing=["./Data/camel/camel-1.6.csv"], configuration_file="./configurations.txt")
    #
    # print "IVY"
    # criteria(name="1", training=["./Data/ivy/ivy-1.1.csv"], tuning=[ "./Data/ivy/ivy-1.4.csv"],
    #      testing=["./Data/ivy/ivy-2.0.csv"], configuration_file="./configurations.txt")
    #
    # print "JEDIT"
    # criteria(name="1", training=["./Data/jedit/jedit-3.2.csv"], tuning=[ "./Data/jedit/jedit-4.0.csv"],
    #      testing=["./Data/jedit/jedit-4.1.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/jedit/jedit-4.0.csv"], tuning=[ "./Data/jedit/jedit-4.1.csv"],
    #      testing=["./Data/jedit/jedit-4.2.csv"], configuration_file="./configurations.txt")
    # criteria(name="3", training=["./Data/jedit/jedit-4.1.csv"], tuning=[ "./Data/jedit/jedit-4.2.csv"],
    #      testing=["./Data/jedit/jedit-4.3.csv"], configuration_file="./configurations.txt")
    #
    # print "LOG4J"
    # criteria(name="1", training=["./Data/log4j/log4j-1.0.csv"], tuning=[ "./Data/log4j/log4j-1.1.csv"],
    #      testing=["./Data/log4j/log4j-1.2.csv"], configuration_file="./configurations.txt")
    #
    # print "LUCENE"
    # criteria(name="1", training=["./Data/lucene/lucene-2.0.csv"], tuning=[ "./Data/lucene/lucene-2.2.csv"],
    #      testing=["./Data/lucene/lucene-2.4.csv"], configuration_file="./configurations.txt")
    #
    # print "POI"
    # criteria(name="1", training=["./Data/poi/poi-1.5.csv"], tuning=[ "./Data/poi/poi-2.0.csv"],
    #      testing=["./Data/poi/poi-2.5.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/poi/poi-2.0.csv"], tuning=[ "./Data/poi/poi-2.5.csv"],
    #      testing=["./Data/poi/poi-3.0.csv"], configuration_file="./configurations.txt")
    #
    # print "SYNAPSE"
    # criteria(name="1", training=["./Data/synapse/synapse-1.0.csv"], tuning=[ "./Data/synapse/synapse-1.1.csv"],
    #      testing=["./Data/synapse/synapse-1.2.csv"], configuration_file="./configurations.txt")
    #
    # print "VELOCITY"
    # criteria(name="1", training=["./Data/velocity/velocity-1.4.csv"], tuning=[ "./Data/velocity/velocity-1.5.csv"],
    #      testing=["./Data/velocity/velocity-1.6.csv"], configuration_file="./configurations.txt")
    #
    # print "XERCES"
    # criteria(name="1", training=["./Data/xerces/xerces-1.1.csv"], tuning=[ "./Data/xerces/xerces-1.2.csv"],
    #      testing=["./Data/xerces/xerces-1.3.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/xerces/xerces-1.2.csv"], tuning=[ "./Data/xerces/xerces-1.3.csv"],
    #      testing=["./Data/xerces/xerces-1.4.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/ant/ant-1.4.csv"], tuning=["./Data/ant/ant-1.5.csv"],
    #          testing=["./Data/ant/ant-1.6.csv"],
    #          configuration_file="./configurations.txt")
    # criteria(name="3", training=["./Data/ant/ant-1.5.csv"], tuning=["./Data/ant/ant-1.6.csv"],
    #          testing=["./Data/ant/ant-1.7.csv"],
    #          configuration_file="./configurations.txt")
    #
    # print "CAMEL"
    # criteria(name="1", training=["./Data/camel/camel-1.0.csv"], tuning=[ "./Data/camel/camel-1.2.csv"],
    #      testing=["./Data/camel/camel-1.4.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/camel/camel-1.2.csv"], tuning=[ "./Data/camel/camel-1.4.csv"],
    #      testing=["./Data/camel/camel-1.6.csv"], configuration_file="./configurations.txt")
    #
    # print "IVY"
    # criteria(name="1", training=["./Data/ivy/ivy-1.1.csv"], tuning=[ "./Data/ivy/ivy-1.4.csv"],
    #      testing=["./Data/ivy/ivy-2.0.csv"], configuration_file="./configurations.txt")
    #
    # print "JEDIT"
    # criteria(name="1", training=["./Data/jedit/jedit-3.2.csv"], tuning=[ "./Data/jedit/jedit-4.0.csv"],
    #      testing=["./Data/jedit/jedit-4.1.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/jedit/jedit-4.0.csv"], tuning=[ "./Data/jedit/jedit-4.1.csv"],
    #      testing=["./Data/jedit/jedit-4.2.csv"], configuration_file="./configurations.txt")
    # criteria(name="3", training=["./Data/jedit/jedit-4.1.csv"], tuning=[ "./Data/jedit/jedit-4.2.csv"],
    #      testing=["./Data/jedit/jedit-4.3.csv"], configuration_file="./configurations.txt")
    #
    # print "LOG4J"
    # criteria(name="1", training=["./Data/log4j/log4j-1.0.csv"], tuning=[ "./Data/log4j/log4j-1.1.csv"],
    #      testing=["./Data/log4j/log4j-1.2.csv"], configuration_file="./configurations.txt")
    #
    # print "LUCENE"
    # criteria(name="1", training=["./Data/lucene/lucene-2.0.csv"], tuning=[ "./Data/lucene/lucene-2.2.csv"],
    #      testing=["./Data/lucene/lucene-2.4.csv"], configuration_file="./configurations.txt")
    #
    # print "POI"
    # criteria(name="1", training=["./Data/poi/poi-1.5.csv"], tuning=[ "./Data/poi/poi-2.0.csv"],
    #      testing=["./Data/poi/poi-2.5.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/poi/poi-2.0.csv"], tuning=[ "./Data/poi/poi-2.5.csv"],
    #      testing=["./Data/poi/poi-3.0.csv"], configuration_file="./configurations.txt")
    #
    # print "SYNAPSE"
    # criteria(name="1", training=["./Data/synapse/synapse-1.0.csv"], tuning=[ "./Data/synapse/synapse-1.1.csv"],
    #      testing=["./Data/synapse/synapse-1.2.csv"], configuration_file="./configurations.txt")
    #
    # print "VELOCITY"
    # criteria(name="1", training=["./Data/velocity/velocity-1.4.csv"], tuning=[ "./Data/velocity/velocity-1.5.csv"],
    #      testing=["./Data/velocity/velocity-1.6.csv"], configuration_file="./configurations.txt")
    #
    # print "XERCES"
    # criteria(name="1", training=["./Data/xerces/xerces-1.1.csv"], tuning=[ "./Data/xerces/xerces-1.2.csv"],
    #      testing=["./Data/xerces/xerces-1.3.csv"], configuration_file="./configurations.txt")
    # criteria(name="2", training=["./Data/xerces/xerces-1.2.csv"], tuning=[ "./Data/xerces/xerces-1.3.csv"],
    #      testing=["./Data/xerces/xerces-1.4.csv"], configuration_file="./configurations.txt")







