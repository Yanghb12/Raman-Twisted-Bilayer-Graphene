import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(0)

x = np.loadtxt(open("./results/confusion_matrix.csv","rb"),delimiter=",", skiprows=0)
print(x)

b = np.sum(x,axis=1)
print(b)
b = b.repeat(4).reshape(4,4) #主要是为了统计每一类的准确率
print(b)

x = x/b
print(x)


f,(ax1) = plt.subplots(figsize=(8,8))

sns.heatmap(x, annot=True, ax=ax1,cmap="YlGnBu", annot_kws={'size':9,'weight':'bold', 'color':'blue'})
# Keyword arguments for ax.text when annot is True.
# http://stackoverflow.com/questions/35024475/seaborn-heatmap-key-words
plt.show()
f.savefig('./test.jpg')
