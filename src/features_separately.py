import matplotlib.pyplot as plt

accuracies = [0.677217928004, 0.691322937824, 0.684630030776, 0.693957037441, 0.688757256379, 0.700518340123, 0.700338038117, 0.679394833249 ]
features = ["allR+B", "allR", "BLUE", "R1", "R2", 'RSU', 'R12', "RSU+BLEU"] #, "R12+BLEU"]
y_axis = range(1, 9)

plt.plot(y_axis, accuracies)
plt.plot(y_axis, accuracies, "bo")
plt.xticks(y_axis, features)
plt.show()