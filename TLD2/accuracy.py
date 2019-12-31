
import matplotlib.pyplot as plt
import pickle
import numpy as np

data_list =[]
with open('./accuracy.pkl','rb') as f:
    while True:
        try:
            data = pickle.load(f)
        except EOFError:
            break
        data_list.append(data)

line = np.ones([3000])
line = line * 0.9

print(np.shape(data_list))
accuracy=np.squeeze(data_list, 0)
plt.title("Average Accuracy per Frames")
plt.xlabel("Frame")
plt.ylabel("Accuracy")
plt.plot(accuracy)
plt.plot(line)
plt.show()
