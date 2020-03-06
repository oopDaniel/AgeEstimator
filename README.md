# Age Estimator

![](./preview/app.png)

Estimate the age based on human facial images. To perform the task, we use 3 different machine learning models:

- Logistic Regression
- Clustering
- Multiclass classification with Convolutional Neural Network

_(NOTE: this project is under development!)_

## Preview

![](./preview/app_uploaded.png)

## Get Started

### Install dependencies

- Server

`pip install --user --requirement requirements.txt`

- Client

`cd app`

`npm i`

### Start the server

In the directory of this repository, do

`python -m server.main &`

### Start the client

`cd app`

`npm run serve &`

## Data

The datasets is composed of 187154 facial images (> 100,000 samples and > 100 features) for training the model. We combine the following 2 datasets and unify their dimensions into 250x250.

- [CACD](https://bcsiriuschen.github.io/CARC/) (160k)
- [UTKFace](https://susanqq.github.io/UTKFace/) (20k)

## Dependencies

- `Python 3.7+`
- `NodeJS 12+`

- Python related: `Numpy`, `Pandas`, `Tensorflow`, `Pytorch`, `CV2`, `PIL`
- JS related: `Vue`, `Ramda`, `Element-UI`, `Font-Awesome`
