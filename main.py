import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("Acquisitions_Eye_tracking_objets_visages_Fix_Seq1/Patients/P_0011_73_M_JT.xlsx", sheet_name="mosob01.jpg - G", header=1)

print(data)

heatmap = np.zeros((64, 64))

length = 1000
width = 1000

for index, row in data.iterrows():
    print(row['x'], row['y'])
    h_x = round(64*(row['x']/length))
    h_y = round(64*(row['y']/width))
    print(h_x, h_y)
    heatmap[h_x][h_y] += 1


for row in heatmap:
    print(row)

plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.show()