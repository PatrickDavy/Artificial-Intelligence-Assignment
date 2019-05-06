import pydotplus
from gplearn.genetic import SymbolicClassifier

with open(r"breast-cancer-wisconsin.data") as classifier_file:
    classifier_data = classifier_file.readlines()

attributes = ["Clump Thickness", "Uniformity of Cell Size",
              "Uniformity of Cell Shape", "Marginal Adhesion",
              "Single Epithelial Cell Size", "Bare Nuclei",
              "Bland Chromatin", "Normal Nucleoli", "Mitoses"]
values = []
alive = []
# Class:

for line in classifier_data:
    data = line.split(',')
    if len(data) > 9:
        temp = []
        for i in range(1, 10):
            x = int(data[i]) if data[i] != "?" else -1
            temp.append(x)

        values.append(temp)

        if int(data[10]) == 2:
            alive.append("benign")
        else:
            alive.append("malignant")


est = SymbolicClassifier(parsimony_coefficient=.01,
                         feature_names=attributes,
                         random_state=10000,
                         verbose=1,
                         stopping_criteria=0.15,
                         population_size=2000,
                         function_set={"mul", "div", "add", "sub", "log"}
                         )

est.fit(values[:400], alive[:400])
print("Accuracy: " + est.score(values[:400], alive[:400]).__str__())
# noinspection PyProtectedMember
# print("Function: " + str(est._program))

# noinspection PyProtectedMember
# graph = pydotplus.graphviz.graph_from_dot_data(est._program.export_graphviz())
# Image(graph.create_png())
# graph.write_png("dtree.png")
