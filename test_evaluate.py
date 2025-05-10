import numpy as np
from utils import calculate_metrics, print_metrics
import csv

def main():
    modality = 'FUSION_1_9'
    rpred_labelsss = []
    bpred_labelsss = []
    test_label = []
    f = open('classification_model_FUSION_1_9/batch_size32/test_subject30_'+modality+'_rprediction.csv','r')
    reader = csv.reader(f)
    for item in reader:
        if reader.line_num == 1:
            continue
        rpred_labelsss.append(item[3])
        test_label.append(item[1])

    test_label = np.array(test_label, dtype='int')
    rpred_labelsss = np.array(rpred_labelsss, dtype='int')
    f.close()

    f = open('classification_model_FUSION_1_9/batch_size32/test_subject30_'+modality+'_bprediction.csv','r')
    reader = csv.reader(f)
    for item in reader:
        if reader.line_num == 1:
            continue
        bpred_labelsss.append(item[3])
    
    bpred_labelsss = np.array(bpred_labelsss, dtype='int')
    f.close()
        
    rmetrics = calculate_metrics(rpred_labelsss, test_label)
    bmetrics = calculate_metrics(bpred_labelsss, test_label)

    print_metrics('resnet', rmetrics)
    print_metrics('bayesian', bmetrics)

if __name__ == '__main__':
    main()