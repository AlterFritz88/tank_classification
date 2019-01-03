import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import  TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, ward
import numpy as np

li_dir = os.listdir(path="truck-link/WWII/СССР")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(li_dir)


linkage_matrix = linkage(X.toarray(), method="complete",metric="euclidean")
dendrogram(linkage_matrix,
           labels=li_dir,
#leaf_rotation=90,
leaf_font_size=8,
orientation = 'left'
)
plt.show()

