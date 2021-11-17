# Influence Maximization with Fairness at Scale - FPS Approach


``` bash
mkdir Code Data Figures
cd Code
git clone <repo location>
```

## Requirements
To run this code you will need the following python packages: 
* [numpy](https://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [scipy](https://www.scipy.org/)
* [tensorflow-gpu](https://www.tensorflow.org/)
* [igraph](https://igraph.org/python/)
* [pyunpack](https://pypi.org/project/pyunpack/)
* [patool](https://pypi.org/project/patool/)

which can be installed using the requirements.txt:

``` bash
pip install -r requirements.txt
```

## Data
All datasets need certain preprocessing before the experiments. 

``` bash
python preprocessing
```
The script creates the required folder structure for the Weibo dataset ->Init_Data,Embeddings, Seeds, Spreading.
It then downloads the [Weibo](https://aminer.org/influencelocality) datasets, and preprocesses them for curation and derivation of the network and the diffusion cascades.<br />



## Run

Run with default parameters for *sampling percentage*, *learning rate*, *number of epochs*, *embeddings size* and *number of negative samples*.

``` bash
python main.py --sampling-perc=120 --learning-rate=0.1 --n-epochs=10 --embedding-size=50 --num-neg-samples=10
```


