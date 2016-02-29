from sklearn.grid_search import GridSearchCV
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


def criteria(training=[],testing=[], configuration_file=None):
    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for train in training:
        contents = get_data(train)
        X_train.extend([c.decisions for c in contents])
        y_train.extend([c.objective for c in contents])

    for test in testing:
        contents = get_data(test)
        X_test.extend([c.decisions for c in contents])
        y_test.extend([c.objective for c in contents])

    # Set the parameters by cross-validation
    tuned_parameters = [{
        'n_estimators': [50, 70,  90,  110, 130, 150],
        'max_features': [0.1,  0.4, 0.7,  1.0],
        'min_samples_split': [1,  9,  17],
        'min_samples_leaf': [2, 4, 6, 8, 10],
        'max_leaf_nodes': [10, 15, 20, 25, 30, 35, 40, 45, 50],
        'random_state': [1]
                         }]

    # Import the random forest package
    from sklearn.ensemble import RandomForestClassifier
    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=2)
    clf.fit(X_train, y_train)

    print "Best parameters set found on development set:"
    print

    y_true, y_pred = y_test, clf.predict(X_test)
    print "Total Number of Evaluations: ", len(clf.grid_scores_)
    print "Best Possible Score of : ", get_F(y_true, y_pred), " can be achieved using : ", clf.best_params_


if __name__ == "__main__":
    criteria(training=["./Data/ant/ant-1.3.csv"], testing=["./Data/ant/ant-1.5.csv"])