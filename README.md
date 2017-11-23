# CNN for Text Classification

This is an implementation of Yoon Kim's "[Convolutional Neural Network for Sentence Classification](https://arxiv.org/abs/1408.5882)" paper in Pytorch.

## Dataset

An dataset is composed of a TSV file and a vocab file. The TSV file has two fields which are separated by a TAB chracter. The first field is a label which should '0' or '1'. (I think this should be changed.) The second field is a sentence that composed of words separated by a whitespace.

The vocab file contains all valid words. A word is in a line.

The MR dataset is provided as a sample. You can find it in ```/dataset/mr/``` directory. There is also an configuration file 'config.yml' describing where the data files is.

## Training

A new experiment can be started by following.

```
python text_classification.py train -c dataset/mr/config.yml conf/small.yml -m ./model
```

Training status can be monitored by tensorboard.

```
tensorboard --logdir ./model
```

An aborted traning session can be resumed by following.

```
python text_classification.py train -m ./model
```

## Prediction

To use a text file for prediction:

```
python text_classification.py predict -m ./model INPUT_FILE
```

Also, ```stdin``` can be used as in input instead of a text file.

```
python text_classification.py predict -m ./model -
```
