
## tinynn


### Basic

tinynn is a lightweight deep learning framework build with pure Python3 and NumPy.

<p align="center">
  <img src="http://ww4.sinaimg.cn/large/006tNc79gy1g63tkgdh1pj30to0fwjsk.jpg" width="80%" alt="tinynn-architecture" referrerPolicy="no-referrer"/>
</p>

The `mini` branch implements the minimal components to run a neural network.


### Getting Started

#### Install

```bash
git clone https://github.com/borgwang/tinynn.git
cd tinynn
pip install -r requirements.txt
```

#### Examples

```bash
cd tinynn
# MNIST classification
python examples/mnist/run.py  
```


#### Components

- layers: Dense
- activation: ReLU, Sigmoid
- losses: SoftmaxCrossEntropy, MSE
- optimizer: Adam


#### License

MIT
