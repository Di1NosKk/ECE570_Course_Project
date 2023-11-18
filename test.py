import numpy as np
import time
import utils
import variables as var
from sklearn.metrics import roc_auc_score 
import LUNAR
import argparse
import model

def main(args):
    # If k is specified, run only once 
    if args.k != 0:
        print(f"Running trial with random seed = {args.seed}")
        print(f"Running trial with k = {args.k}")
        train_x, train_y, val_x, val_y, test_x, test_y = utils.load_data(args.dataset,args.seed)

        start = time.time()
        if args.models == 1:
            test_out = model.run(train_x,train_y,val_x,val_y,test_x,test_y,args.dataset,args.seed,args.k,args.samples,args.train_new_model, args.plot)
        elif args.models == 0:
            test_out = LUNAR.run(train_x,train_y,val_x,val_y,test_x,test_y,args.dataset,args.seed,args.k,args.samples,args.train_new_model)
        end = time.time()
        # print(test_y)
        # print(test_out)

        score = 100*roc_auc_score(test_y, test_out)

        print('Dataset: %s \nSamples: %s \t Score: %.4f \nRuntime: %.2f seconds' %(args.dataset,args.samples,score,(end-start)))

    # If k is not specified, run all k with the same seed
    else:
        for k in [2,50,100,150,200,300]:
    
            print(f"Running trial with random seed = {args.seed}")
            print(f"Running trial with k = {k}")
            train_x, train_y, val_x, val_y, test_x, test_y = utils.load_data(args.dataset,args.seed)
    
            start = time.time()
            if args.models == 1:
                test_out = model.run(train_x,train_y,val_x,val_y,test_x,test_y,args.dataset,args.seed,k,args.samples,args.train_new_model, args.plot)
            elif args.models == 0:
                test_out = LUNAR.run(train_x,train_y,val_x,val_y,test_x,test_y,args.dataset,args.seed,k,args.samples,args.train_new_model)
                
            end = time.time()     
    
            score = 100*roc_auc_score(test_y, test_out)
    
            print('Dataset: %s \nSamples: %s \t Score: %.4f \nRuntime: %.2f seconds' %(args.dataset,args.samples,score,(end-start)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help = "DATASET: LYMPHO, MAMMOGRAPHY, MNIST, MUSK, PENDIGITS, SHUTTLE, THYROID, SATELLITE")
    parser.add_argument("--samples", type = str, help = "UNIFORM, SUSPACE, MIXED")
    parser.add_argument("--k", type = int, default = 0, help = "2, 10, 50, 100, 150, 200, or more")
    parser.add_argument("--seed", type = int, default = 42, help = "Enter any random seed")
    parser.add_argument("--train_new_model", action="store_true", help = 'Train a new model vs. load existing model')
    parser.add_argument("--plot", action="store_true", help = 'Plot if want to visualize training')
    parser.add_argument("--models", type = int, default = 1, help = 'Run with LUNAR model: 0 for Original model, 1 for Our model')
    args = parser.parse_args()

    main(args)