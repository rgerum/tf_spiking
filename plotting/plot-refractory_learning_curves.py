import numpy as np
import matplotlib.pyplot as plt
import os

source_folder = r'C:\Projekte\lif_refractory\training\logs\color_3Lifs_30LSTM0.5_refractory_test'
folders = next(os.walk(source_folder))[1]
#print(folders)


experiments = ['pre_dense_layer_False_train_LIF_False', 'pre_dense_layer_False_train_LIF_True', 'pre_dense_layer_True_train_LIF_False', 'pre_dense_layer_True_train_LIF_True']
short_exp = ['FalseFalse', 'FalseTrue', 'TrueFalse', 'TrueTrue']

folder_list_experiment = {}
for i, experiment in enumerate(experiments):
    folder_list_experiment[short_exp[i]] = []
    for folder in folders:
        if experiment in folder:
            #print('folder_name', folder)
            folder_list_experiment[short_exp[i]].append(folder)
    print(folder_list_experiment[short_exp[i]])
    print('###')


print(folder_list_experiment)

fig = plt.figure(figsize = (10,10))

for j, short in enumerate(short_exp):
    ax = fig.add_subplot(2,2,j+1)
    for n, x in enumerate(folder_list_experiment[short]):
        data = np.loadtxt(os.path.join(source_folder, x, 'data.txt'))
        print(x)
        ax.plot(np.arange(1, len(data[:, 3]) + 1), data[:, 3], color = [0, n/10, (10-n)/10], lw = 2, label = x[0:9])
        ax.set_ylim([0.05,0.55])
        ax.set_xlim([0, 200])
    ax.set_title(experiments[j])
ax.legend()
plt.show()