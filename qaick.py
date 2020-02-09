# -*- coding: utf-8 -*-
import pandas
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNC
import os

def aicreate(csv_path, feature1, feature2, label_column, labels, k = 5, x_test = False, y_test = False, graphical_view = True, save_path = os.path.dirname(os.path.realpath(__file__))):
    print("load data...")
    csv = pandas.read_csv(csv_path)
    x = csv.loc[:, feature1]
    y = csv.loc[:, feature2]
    lab = csv.loc[:, label_column]

    if graphical_view:
        print("make the graphic...")
        plt.scatter(x[lab == 0], y[lab == 0], color = 'g', label = labels[0])
        plt.scatter(x[lab == 1], y[lab == 1], color = 'r', label = labels[1])
        plt.scatter(x[lab == 2], y[lab == 2], color = 'b', label = labels[2])
        plt.scatter(x_test, y_test, color='k')

    print("create model...")
    d = list(zip(x, y))
    model = KNC(n_neighbors = k)
    model.fit(d, lab)

    if graphical_view:
        print("save the graphic...")
        path = os.path.join(save_path, 'graph.png')
        plt.savefig(path)

    if x_test != False or y_test != False:
        print("test...")
        return show_results(model, x_test, y_test, labels)

def show_results(model, x_test, y_test, labels):
    print("predict...")
    prediction = model.predict([[x_test, y_test]])

    print("create returned sentence...")
    result = "Résultats : "
    if prediction[0] == 0:
        result += labels[0]
    elif prediction[0] == 1:
        result += labels[1]
    elif prediction[0] == 2:
        result += labels[2]

    return result

"""
required arguments:
    • csv_path: full path to access to the csv file
    • feature1: the name of one of the column in your csv file which is a feature for your artificial intelligence; it represents the x axis
    • feature2: the name of one of the column in your csv file which is a feature for your artificial intelligence; it represents the y axis
    • label_column: the name of one of the column in your csv file where there are the labels of each lines
    • labels: every labels in list form
optionnal arguments:
    • k: the number of the nearest neighbors
    • x_test: if you want test with a x coordonates
    • y_test: [required if x_test] if you want test with a y coordonates
    • graphical_view: if you want save the graphic in a scatterplot form
    • save_path: the full path of the folder where you want to save the graph, if left as is, corresponds to the path of the executed file

example:
    aicreate(csv_path = "/Users/user/Documents/ai/iris.csv", feature1 = "petal_length", feature2 = "petal_width", label_column = "species", labels = ["setosa", "virginica", "versicolor"], x_test = 1.3, y_test = 0.3)
"""
