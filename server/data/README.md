# Dataset


## Decompress the raw data

To unzip the images, run the following script:

`cat dataset.tar.* | tar -xzvf -`

## Get Started


### Load data into Python

In your model, do

```python
from server.data.dataset import DataLoader

dl = DataLoader()
x_train, y_train = dl.load_train()
x_test, y_test = dl.load_test()
```

to start playing with the data.

*Note: the training data are strings of file name, so you'll still need to load the file as matrix to perform any scientific computation.*

### Load image by file name

There are several ways to load an image. One way is use `matplotlib`:

```python
import matplotlib.image as img

image = img.imread(x_train[0])
```
