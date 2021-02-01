# Meta-Learning Adversarial Domain Adaptation Network for Few-Shot Text Classification

## Data
#### Download

We ran experiments on a total of 4 datasets. You may download our processed data [here](https://people.csail.mit.edu/yujia/files/distributional-signatures/data.zip).

| Dataset | Notes |
|---|---|
| 20 Newsgroups ([link](http://qwone.com/~jason/20Newsgroups/ "20 Newsgroups")) | Processed data available. We used the `20news-18828` version, available at the link provided.
| Reuters-21578 ([link](https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.html "Reuters")) | Processed data available. |
| Amazon reviews ([link](http://jmcauley.ucsd.edu/data/amazon/ "Amazon")) | We used a subset of the product review data. Processed data available. |
| HuffPost&nbsp;headlines&nbsp;([link](https://www.kaggle.com/rmisra/news-category-dataset "HuffPost")) | Processed data available. |

Please download `wiki.en.vec` from [here](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec) and put it under `pretrain_wordvec` folder.

#### Format

- Each JSON file contains one example per line. With the exception of RCV1, each example has keys `text` and `label`. `text` is a list of input tokens and `label` is an integer, ranging from 0 to the number of classes - 1.
- For RCV1, we are unable to distribute the original data. In place of input tokens, each example specifies a `path`, which corresponds to each example's file path in the data distribution.
- Class splits for each dataset may be found in `src/dataset/loader.py`.

### Quickstart
Run our model with default settings. By default we load data from `data/`.
```
cd bin
sh mlada.sh
```

## Code
`src/main.py` may be run with one of three modes: `train`, `test`, and `finetune`.
- `train` trains the meta-model using episodes sampled from the training data.
- `test` evaluates the current meta-model on 1000 episodes sampled from the testing data.
- `finetune` trains a fully-supervised classifier on the training data and finetunes it on the support set of each episode, sampled from the testing data.


#### Dependencies
- Python 3.7
- PyTorch 1.6.0
- numpy 1.18.5
- torchtext 0.7.0
- termcolor 1.1.0
- tqdm 4.46.0
- CUDA 10.2
