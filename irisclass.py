import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import tree
import pandas
import graphviz 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pandas.read_csv(url, names=names)
iris = datasets.load_iris()
print(data.describe())
data.hist()
plt.show()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")
