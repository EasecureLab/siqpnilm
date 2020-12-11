import pandas as pd
from libDataLoaders import dataset_loader
import collections
import nilm
from ikhmm import *
from evaluator import *

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
hmms = collections.OrderedDict()
app_ids = list(data)[1:-1]
for app_id in app_ids:
    hmm = IterativeKmeansHMM(data[app_id].head(3 * 1440), max_k=4, std_thres=1)
    hmm.fit()
    hmms[app_id] = hmm
solver = nilm.SIQP(aggregate, hmms=hmms, step_thr=2)
solver.solve()
ground_truth = data[app_ids].tail(36 * 1440)
evaluator = Evaluator(ground_truth, solver.estimate, solver.aggregate)
evaluator.show()
print(evaluator.report)
