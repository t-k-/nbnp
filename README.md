<p align="center">
<img src="https://i.loli.net/2019/09/24/VatMxSiXLdg8yrD.png"/>
</p>

## About

nbnp is forked from tinynn, a lightweight neural network framework designed with simplicity in mind. It also aim for educational purpose, to help people get a high level but detailed technical picture of neural network or deep learning.
It is currently written in Python 3 using Numpy.

## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Examples

```bash
cd tinynn
# MNIST classification
python examples/mnist/run.py  
# a toy regression task
python examples/nn_paint/run.py  
 # reinforcement learning demo (gym environment required)
python examples/rl/run.py
```

## Contribute

Please follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for Python coding style.

In addition, please sort the module import order alphabetically in each file. To do this, one can use tools like [isort](https://github.com/timothycrosley/isort) (be sure to use `--force-single-line-imports` option to enforce the coding style).

## License

MIT
