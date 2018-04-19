import matplotlib.pyplot as plt

#RF
accuracies = [0.69005, 0.69045, 0.69710, 0.67747, 0.64711, 0.68781, 0.69540, 0.69973, 0.62647, 0.63634, 0.66866 ]
features = ["R1", "R2", "RL", "RS4", "BLUE", "R", "R12L", "keywords_RL", "keywordsSor", "keywordsJac", "RLkeywordsJac"  ]


y_axis = range(1, 12)

plt.plot(y_axis, accuracies)
plt.plot(y_axis, accuracies, "bo")
plt.xticks(y_axis, features)
plt.show()