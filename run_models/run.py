ROOT_DIR = '/gpfs/commons/groups/gursoy_lab/aelhussein/ot_cost/otcost_fl_rebase'
import sys
sys.path.append(f'{ROOT_DIR}/code/helper')
import pipeline as pp
import hyperparameter_search_pipeline as hp
import models_helper as mh
import trainers as tr
import data_preprocessing as dp
import importlib
importlib.reload(pp)
importlib.reload(hp)
importlib.reload(mh)
importlib.reload(tr)
importlib.reload(dp)
import argparse


costs_dict = {'Synthetic': [0.03, 0.10, 0.20, 0.30, 0.40, 0.50],
              'Credit': [0.12, 0.23, 0.30, 0.40],
              'Weather': [0.11, 0.19, 0.30, 0.40, 0.48],
              'EMNIST': [0.11, 0.19, 0.25, 0.34, 0.39],
              'CIFAR': [0.08, 0.21, 0.3, 0.38], 
              'IXITiny': [0.08, 0.28, 0.30],
              'ISIC': [0.06, 0.15, 0.19, 0.25, 0.3]
              } 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ds", "--dataset")
    parser.add_argument("-hp", "--hyperparameter", default = 'False')
    args = parser.parse_args()
    DATASET = args.dataset
    costs = costs_dict[DATASET]
    if args.hyperparameter == 'False':
        results_scores, results_train_losses, results_val_losses, results_test_losses = pp.runAnalysis(DATASET, costs)
    else:
        results_scores = hp.runAnalysis(DATASET, costs)
    
if __name__ == "__main__":
    main()