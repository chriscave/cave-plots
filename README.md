# Feynplots
Feynplots is a package to visualise each interaction in a Feyn Graph. See [here for more on starting with Feyn](https://docs.abzu.ai/docs/guides/quick_start.html) and [understanding the QLattice.](https://docs.abzu.ai/docs/guides/qlattice.html)

## Install
This can be installed from [PyPi.](https://pypi.org/project/feynplots/)
```
richard@feyn:~$ pip install feyn
```


## About
Every interaction in a Feyn Graph has either one or two input variables. This means that every interaction can be represented in a two dimensional plot.

When an interaction has one input then the axis of the plot is the input on the x-axis and the output on the y-axis.

When an interaction has two inputs then the axes of the plot are both inputs and then the contour lines represent the output of the interaction

The purpose of this package is to return a figure that contains every plot of an interaction in a Feyn Graph.

##Use
The following demonstrates how to use the package in a full workflow with the QLattice. We use the California housing dataset as an example.
```python
import feyn
from feyn import tools
from feynplots import GraphPlot

import sklearn.datasets
import pandas as pd
import numpy as np

#Importing the dataset from sklearn and turning it into a Pandas DataFrame
dataset = sklearn.datasets.fetch_california_housing()
data = pd.DataFrame(data=dataset.data,columns=dataset.feature_names)
price = pd.DataFrame(data=dataset.target, columns = ['price'])
cal_housing = pd.concat([data,price],axis=1)

#Splitting into a train and test set
train, test = tools.split(cal_housing,ratio=[4,1])

#Here you would input your url and api token
ql = feyn.QLattice(url='Your unique url here', api_token='Your unique api token here')
ql.reset()

#Adding registers
for cols in cal_housing.columns:
    ql.get_register(name=cols)
    
#Extract, fitting, updating and repeating
updates = 10
epochs = 15
target = 'price'
for loop in range(updates):
    qgraph = ql.get_qgraph(train.columns,target,max_depth=3)
    qgraph.fit(train,epochs = epochs,threads = 5)
    best = qgraph.select(train,n=1)[0]
    ql.update(best)
    
best = qgraph.select(train, n = 1)[0]

#Using the package feynplots. First initiates the instance with a Feyn graph
graphplot = GraphPlot(best) 

#evaluates every interaction at every datapoint in a pandas DataFrame
graphplot.model_ev(train)

#plots the figure the includes every interaction
graphplot.plot(figsize = (30,20)) 
```
##Analysis of plots
Here we explain the output of the function GraphPlot.plot(). It is a matplotlib figure that contains every plot of every interaction in the Feyn Graph. We will start off with an example of a Feyn Graph: 

```python
example = ql.select(train, n = 1)[0]
example
```





an example of a Feyn graph


A package to see what's going on inside graphs produced from a QLattice. A QLattice is a quantum mechanics simulator produced by [Abzu](https://www.abzu.ai/) that produces models for datasets in an evolutionary process. [You can read more about the QLattice here](https://docs.abzu.ai/docs/guides/qlattice.html). Feyn is the package used to interact with the QLattice and you can find more about [getting started with it here](https://docs.abzu.ai/docs/guides/quick_start.html).

A graph produced from the QLattice typically looks like so



This is a model for the California housing dataset. Each interaction takes either one or two variables as input and has a single output. This means that we can plot each interaction. Let's do that!


Here's a couple of comments on this plot
Each dot corresponds to a datapoint in the training set. The colour corresponds to the actual value of the target variable.
The x-axis corresponds to the variable x0;
The y-axis corresponds to the variable x1;
The scale on each axis the scale of each feature;
The contour lines correspond to the value of the output at the (x0,x1) coordinate.

Here's a small summary of how to use this package.

