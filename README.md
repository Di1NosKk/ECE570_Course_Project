# ECE570_Course_Project

## Running the experiment:

Running environment: Google Colab

1. Install the necessary Python packages

Run:

```
pip install faiss-gpu
pip install torch_geometric
```

2. Download the datasets, [PENDIGITS/SATELLITE/SHUTTLE/THYROID/others](http://odds.cs.stonybrook.edu)

3. Modify  the file path in `load_data()` from `utils.py`

4. Use `starting_engine.ipynb` to start to access Google Colab

5. Open the terminal and run:

```
python test.py --dataset MNIST --samples MIXED --k 100 --seed 42 --train_new_model --models 1 --plot
```

If you need some help with running the `test.py`, then run:
```
python test.py -h
python test.py --help
```
The output will list all the options for helping instruction. 
![Alt text](https://github.com/Di1NosKk/ECE570_Course_Project/blob/master/helping.png)

## Description of files 
Note: Official Implementation of ["LUNAR: Unifying Local Outlier Detection Methods via Graph Neural Networks"](https://www.aaai.org/AAAI22Papers/AAAI-51.GoodgeA.pdf), Adam Goodge, Bryan Hooi, Ng See Kiong and Ng Wee Siong (AAAI2022) and Code Reference is [here](https://github.com/agoodge/LUNAR)

`LUNAR.py`, `appendix.pdf`, and `utils.py` are copied from the official implementation

`variables.py`: Initialize some global variables for using training and evaluating

`starting_engine.ipynb`: To install and start a Python environment in Google Colab

`test.py`: Copied from the repo and added a few parameters for implementation. The helping instruction is shown above. 

The parameters `k`, `seed`, `train_new_model`, `models`, `plot` are optional.  

Default value:
1. `k`: to run all the k if k is not specified, default = 0
2. `seed`: to specify a seed for running, default = 42
3. `train_new_model`: whether or not to train the model, default = false
4. `models`: 0 for the models from the repo, 1 for our reimplemented model, default = 1
5. `plot`: to plot the train loss and score curve, default = false

`model.py`: build a GNN model and run training and evaluation, GNN output layer has `logsigmoid()` activate function. Implement the plot feature in this file and save it in the local directory. 

`test.py`: running the model with two situations, if k is specified then run once, else if k is not specified then run all the k. If the model is specified then run the specified model, else if the model is not specified then the default is to run our LUNAR model

`train.py`: train the model and save the loss value list and score value list as well as the best model. Using all the outputs for our plotting feature and evaluation

`evaluate.py`: save the output using the trained model and loss value

## Dataset Descriptions

[Datasets](http://odds.cs.stonybrook.edu)

All the datasets we using are one record per data point, and each record contains several attributes. From the website that provides the datasets, we can see the size of each data set and the proportion of outliers it contains.  For example in MNIST, it is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. This dataset is converted for outlier detection as digit-zero class is considered as inliers, while 700 images are sampled from digit-six class as the outliers. In addition, 100 features are randomly selected from 784 total features.
