import time
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from detectors import EEIF
import argparse


File_name = ['arr']
ap = argparse.ArgumentParser()
output_data = []
ap.add_argument("-IntB", "--threshold", required=True, help="threshold")
args = vars(ap.parse_args())

threshold = int(args["threshold"])
branch = 0

num_ensemblers = 100

for filename in File_name:
    print("Dataset:",filename)
    glass_df = pd.read_csv('data/' + filename + '.csv', header=None)  
    glass_df = glass_df.astype(float)
    X = glass_df.values[:, :-1]  
    ground_truth = glass_df.values[:, -1]

    detectors = [("L2OPT", EEIF('L2OPT', num_ensemblers, threshold, branch))] 
    for i, (dtc_name, dtc) in enumerate(detectors):
        print("\n" + dtc_name + ":")
        AUC = []
        PR_AUC = []
        Traintime = []
        Testtime = []
        for j in range(15):
            print(j,'\n')
            start_time = time.time()
            dtc.fit(X)
            train_time = time.time() - start_time
            y_pred = dtc.decision_function(X)
            test_time = time.time() - start_time - train_time
            auc1 = roc_auc_score(ground_truth, -1.0*y_pred)
            AUC.append(auc1)
            pr_auc = average_precision_score(ground_truth, -1.0*y_pred)
            PR_AUC.append(pr_auc)
            Traintime.append(train_time)
            Testtime.append(test_time)
        mean_auc = np.mean(AUC)
        std_auc = np.std(AUC)
        mean_pr = np.mean(PR_AUC)
        std_pr = np.std(PR_AUC)
        mean_traintime = np.mean(Traintime)
        std_traintime=np.std(Traintime)
        mean_testtime = np.mean(Testtime)

        print("\tAUC score:\t", mean_auc)
        print("\tAUC std:\t", std_auc)
        print("\tPR score:\t", mean_pr)
        print("\tPR std:\t", std_pr)
        print("\tTraining time:\t", mean_traintime)
        print("\tstd_traintime:\t", std_traintime)
        print("\tTesting time:\t", mean_testtime)
        
        output_data.append([filename, np.mean(AUC), np.std(AUC), np.mean(PR_AUC), np.std(PR_AUC), np.mean(mean_traintime), np.std(mean_traintime), np.mean(mean_testtime), np.std(mean_testtime)])