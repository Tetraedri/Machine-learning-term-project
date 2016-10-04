import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

#df = pd.read_csv('classification_dataset_training.csv', delimiter=',', encoding="utf-8-sig")

my_data = np.genfromtxt('classification_dataset_training.csv', delimiter=',', dtype=None)
positives = [a for a in my_data if a[len(my_data[0])] not in [0]]
print(positives)
result = [a for a in A if a not in subset_of_A]



print(len(my_data[1][0]))
