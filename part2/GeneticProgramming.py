import pydotplus
from gplearn.genetic import SymbolicRegressor

with open(r"regression.txt") as regression_file:
    regression_data = regression_file.readlines()

x = []
y = []
i = 0
for line in regression_data:
    if i == 0:
        i += 1
        continue
    locations = line.split()
    if len(locations) == 2:
        x.append([float(locations[0])])
        y.append(float(locations[1]))
    else:
        continue

est_gp = SymbolicRegressor(population_size=5000,
                           generations=15,
                           stopping_criteria=0.01,
                           p_crossover=0.7,
                           p_subtree_mutation=0.1,
                           p_hoist_mutation=0.08,
                           p_point_mutation=0.1,
                           max_samples=0.9,
                           verbose=1,
                           parsimony_coefficient=0.01,
                           random_state=50)

est_gp.fit(x, y)
print("Accuracy: " + str(est_gp.score(x, y)*100) + "%")

print("Function: " + str(est_gp))

# noinspection PyProtectedMember
# graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
# Image(graph.create_png())
# graph.write_png("dtree.png")
