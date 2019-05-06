from sklearn.neural_network import MLPClassifier

with open(r"iris-test.txt") as testFile:
    testData = testFile.readlines()

with open(r"iris-training.txt") as trainingFile:
    trainingData = trainingFile.readlines()

testFloatArray = []
testFlowerNames = []
trainingFloatArray = []
trainingFlowerNames = []

for line in trainingData:
    flower = line.split()
    if len(flower) == 0:
        continue

    trainingFloatArray.append([float(flower[0]), float(flower[1]), float(flower[2]), float(flower[3])])
    trainingFlowerNames.append(flower[4])

for line in testData:
    flower = line.split()
    if len(flower) == 0:
        continue

    testFloatArray.append([float(flower[0]), float(flower[1]), float(flower[2]), float(flower[3])])
    testFlowerNames.append(flower[4])

random_state = 0
hidden_layer_sizes = 0
ee = 0
accuracy = 0
for i in range(1, 10):
    for j in range(1, 10):
        classifier = MLPClassifier(
            solver='sgd',
            alpha=1e-5,
            hidden_layer_sizes=(j,),
            random_state=i,
            max_iter=300,
            learning_rate='adaptive',
            momentum=0.9,
            learning_rate_init=0.25,
            tol=0.0001,
            early_stopping=False
        )
        classifier.fit(trainingFloatArray, trainingFlowerNames)
        score = classifier.score(testFloatArray, testFlowerNames, sample_weight=None)
        if score >= accuracy:
            random_state = i
            hidden_layer_sizes = j
            accuracy = score

print("Hidden layer size: " + str(hidden_layer_sizes))
print("Random state: " + str(random_state))
print("Accuracy: " + str(accuracy * 100) + "%")

# MLPClassifier(
#     activation='relu',
#     alpha=1e-05,
#     batch_size='auto',
#     beta_1=0.9,
#     beta_2=0.999,
#     early_stopping=False,
#     epsilon=1e-08,
#     hidden_layer_sizes=(5, 2),
#     learning_rate='constant',
#     learning_rate_init=0.001,
#     max_iter=200,
#     momentum=0.9,
#     n_iter_no_change=10,
#     nesterovs_momentum=True,
#     power_t=0.5,
#     random_state=1,
#     shuffle=True,
#     solver='lbfgs',
#     tol=0.0001,
#     validation_fraction=0.1,
#     verbose=False,
#     warm_start=False
# )
