import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# A tiny LSA example for the following sentence set: {("The sun is bright.", "S1"), ("Such a bright, bright student!", "S2"),
#  ("Lying student is lying in the sun.", "S3"), ("That's a lie!", "S4")}

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
matrix = np.array([[1, 0, 1, 0], [1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 2, 1]])
u, s, vh = np.linalg.svd(matrix)
print("U = " + str(u))
print("Sigma = " + str(s))
print("V* = " + str(vh))
#keep only 2d:
s = np.array([[2.46, 0, 0, 0], [0, 1.7, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
u = np.column_stack((u[:,0], u[:,1]))
vh = np.row_stack((vh[0],vh[1]))
plotting = np.row_stack((u, vh.T))
lables = ("sun", "bright", "student", "lie", "S1", "S2", "S3", "S4")
print(plotting[0])
#plt.plot(plotting[:,0], plotting[:,1], "bo")


k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
k_means.fit(plotting)
colors = {0 : "bo", 1:"ro", 2:"go", 3:"yo"}
k_means_labels = k_means.labels_
centers = k_means.cluster_centers_
print(centers)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
for i, label in enumerate(lables):
    plt.text(plotting[i,0], plotting[i,1], label)
    plt.plot(plotting[i, 0], plotting[i, 1], colors[k_means_labels[i]])
plt.plot(centers[:,0], centers[:,1], "ko")
plt.show()