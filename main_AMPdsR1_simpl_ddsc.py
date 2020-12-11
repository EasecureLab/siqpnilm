## 单电器--字典学习，稀疏编码-->预测-->通过fhmm框架呈现结果-->数据集为AMPdsR
## 遗留问题
## 1、多电器
## 2、用原生的字典，控制约束更方便

import pandas as pd
from libDataLoaders import dataset_loader
import collections
import nilm
from ikhmm import *
from evaluator import *
from ddsc import *
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from time import time
from sklearn.metrics import mean_squared_error

def print_appliance_wise_errors(activations, bases, compTemp):
    start_comp = 0
    power = train_data
    # for cnt, i in enumerate(power):
    X = power
    n_comps = compTemp
    pred = np.matmul(bases[:, start_comp:start_comp + n_comps], activations[start_comp:start_comp + n_comps, :])
    start_comp += n_comps
    print("Sparse Coding MSE Error for  appliance  is", mean_squared_error(pred, X) ** (.5))

pd.set_option('display.max_columns', None)
model_args = pd.read_csv("model_building_args.csv", header=None)
n_appliance = 1
args = model_args.loc[n_appliance - 1, :]
dataset = args[1]
precision = args[2]
denoised = args[4]
ids = args[7].split(',')
datasets_dir = './data/%s.csv'
data = dataset_loader(datasets_dir % dataset, ids, precision=precision, denoised=denoised)
aggregate = data.WHE.tail(36 * 1440)
# train data for dictionary learning
train_data = data.tail(1440)
model = MiniBatchDictionaryLearning(n_components=100, alpha=1, n_iter=30)
model.fit(train_data.T)
comp = model.components_
basics = comp.T

n_componetss = model.n_components
transforms = model.transform(train_data.T)
activations = transforms.T
reconstruction = np.matmul(basics, activations)
print("Dictionary Learning RMSE for appliance  is %s" % (mean_squared_error(reconstruction, train_data) ** (.5)))
## SparseCoder for appliance
model = SparseCoder(dictionary=basics.T, positive_code=True, transform_algorithm='lasso_lars')
predicted_activations = model.transform(train_data.T).T
print_appliance_wise_errors(predicted_activations, basics, n_componetss)

start_comp1 = 0
predxx =np.dot(basics[:, start_comp1:start_comp1 + n_componetss], activations[start_comp1:start_comp1 + n_componetss, 1])
predxx = pd.DataFrame(predxx)
app_ids = list(data)[1:-1]
ground_truth = data[app_ids].tail(1440)
evaluator = Evaluator(ground_truth, predxx, train_data)
evaluator.show()
print(evaluator.report)
