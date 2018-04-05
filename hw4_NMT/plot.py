import matplotlib.pyplot as plt
#fig = plt.figure()
#
# k = [1,2,3,4]
# train_256 =[55.1919, 45.7523, 49.8140, 53.0738]
# train_512 = [58.2446, 43.7816, 47.0434, 30.2829]
# train_1024 = [50.5634, 40.7006, 42.2953, 22.8020]
# train_2048 = [50.3951, 40.9001, 29.0785, 16.6689]
#
# plt.plot(k,train_256,linestyle='-',color='b',label='256')
# plt.plot(k, train_512, linestyle='-',  color='r', label='512')
# plt.plot(k, train_1024, linestyle='-',  color='g', label='1024')
# plt.plot(k, train_2048, linestyle='-',  color='y', label='2048')
# plt.legend(loc='up right')
# plt.xlabel('Epochs')
# plt.ylabel('Training Error')
# plt.title("Traing Error vs Epochs for different number of hidden nodes")
# plt.show()
# print ("hi")
#
# dev_256 = [57.5523, 55.9347, 55.9765, 56.3921]
# dev_512 = [53.6744, 51.3875, 51.4436, 52.6610]
# dev_1024 = [54.5805, 53.6588, 56.8827, 62.1201]
# dev_2048 = [53.6515, 53.1403, 58.9789, 66.2928]
# plt.plot(k,dev_256,linestyle='-',color='b',label='256')
# plt.plot(k, dev_512, linestyle='-',  color='r', label='512')
# plt.plot(k, dev_1024, linestyle='-',  color='g', label='1024')
# plt.plot(k, dev_2048, linestyle='-',  color='y', label='2048')
# plt.legend(loc='up right')
# plt.xlabel('Epochs')
# plt.ylabel('Dev Error')
# plt.title("Dev Error vs Epochs for different number of hidden nodes")
# plt.show()
# print ("hi")


# k	 = [4e-5	 ,   8e-5	,  4e-4	  , 8e-4]
# Train_Loss	 = [46.5441,	  41.3048,	 42.2168,	 47.7085]
# Dev_Loss	 = [52.4803	  ,52.4297	, 51.2689	, 51.5785]
# # Test_BLEU	  = [ 0.0727,	    0.0760	  ,0.0744	,  0.0732]
# plt.plot(k,Train_Loss,linestyle='-',color='b',label='Train Loss')
# plt.plot(k, Dev_Loss, linestyle='-',  color='r', label='Dev Loss')
# # plt.plot(k, Test_BLEU, linestyle='-',  color='g', label='Test BLEU')
# plt.legend(loc='up right')
# plt.xlabel('Regularization Strength')
# plt.ylabel('Error')
# plt.title("Train Error and Dev Error vs Regularization Strength")
# plt.show()
# print ("hi")

import numpy as np

A = [[0.0601,0.0750,0.0704],[0.0798,0.0688,0.0694]]
A = np.array(A)
import matplotlib.ticker as ticker

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(A)
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([0,1,2,3])
ax.set_yticklabels([0,512,1024])

# # Show label at every tick
# ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
plt.show()
plt.close()

# def plot_confusion_matrix(cm,cls,normalize=False):
#     import matplotlib.pyplot as plt
#     cmap = plt.cm.Blues
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm,interpolation='nearest',cmap=cmap)
#     plt.title("Logistic Reg for NER with Features")
#     plt.colorbar()
#     tick_marks = np.arange(len(cls))
#     plt.xticks(tick_marks, cls,rotation = 90)
#     plt.yticks(tick_marks, cls)
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     fmt = '.1f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     import itertools
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.show()
#     plt.savefig("LOG_NER_BASE.png")


