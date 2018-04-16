import matplotlib.pyplot as plt

accuracies = [0.69005, 0.69045, 0.69710, 0.67747, 0.64711, 0.68781, 0.69540, 0.69973 ]
features = ["R1", "R2", "RL", "RS4", "BLUE", "R", "R12L", "keywords_R12L"  ]


y_axis = range(1, 8)

plt.plot(y_axis, accuracies)
plt.plot(y_axis, accuracies, "bo")
plt.xticks(y_axis, features)
plt.show()