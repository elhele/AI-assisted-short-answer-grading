import matplotlib.pyplot as plt


accuracies = [0.677217928004, 0.691322937824, 0.684630030776, 0.693957037441, 0.688757256379, 0.700518340123, 0.700338038117, 0.679394833249 ]
x_values = ["allR+B", "allR", "BLUE", "R1", "R2", 'RSU', 'R12', "RSU+BLEU"] #, "R12+BLEU"]
plt.xlabel('Feature combination')
plt.ylabel('Average accuracy')
plt.grid(True)
plt.plot(x_values, accuracies)
plt.plot(x_values, accuracies, 'bo')
plt.show()

a = [1,2,3]
print(a[:2])