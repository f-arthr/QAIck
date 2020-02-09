# QAIck
QAIck is a module to make and, optionnaly, test a simple artificial intelligence

## How use it?

Example:
```python
aicreate(csv_path = "/Users/user/Documents/ai/iris.csv", feature1 = "petal_length", feature2 = "petal_width", label_column = "species", labels = ["setosa", "virginica", "versicolor"], x_test = 1.3, y_test = 0.3)
```

Required arguments:
* `csv_path`: full path to access to the csv file
* `feature1`: the name of one of the column in your csv file which is a feature for your artificial intelligence; it represents the x axis
* `feature2`: the name of one of the column in your csv file which is a feature for your artificial intelligence; it represents the y axis
* `label_column`: the name of one of the column in your csv file where there are the labels of each lines
* `labels`: every labels in list form

Optionnal arguments:  
* `k`: the number of the nearest neighbors
* `x_test`: if you want test with a x coordonates
* `y_test:` [required if x_test] if you want test with a y coordonates
* `graphical_view`: if you want save the graphic in a scatterplot form
* `save_path`: the full path of the folder where you want to save the graph, if left as is, corresponds to the path of the executed file
