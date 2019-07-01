# YANN (Yet Another Neural Network library)

YANN is the Neural Network library implemented from ground up using
the [Eigen](https://eigen.tuxfamily.org) library for matrix operations.
The code uses many C++17 features (it was compiled and tested using
g++ 8.3 and should compile on most modern compilers).

### Features
- Fully connected networks, CNN, RNN, LSTM, ...
- Gradient descent, AdaGrad, AdaDelta, ...
- Sigmoid, Tanh, Quadratic Cost, Entropy Cost, ...

### Installing

Install [Boost](http://boost.org/) and [Eigen](https://eigen.tuxfamily.org),
for example on Ubuntu:

```
sudo apt install libboost-all-dev libeigen3-dev zlib1g-dev
```

Optionally, install [Intel MKL](https://software.intel.com/en-us/mkl)
(see [Eigen docs](https://eigen.tuxfamily.org/dox/TopicUsingIntelMKL.html) for
more details):

```
cd /tmp
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

sudo sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
sudo apt-get update && sudo apt-get install intel-mkl
```

Followed by the usual

```
./configure && make && make check
```

The two helper scripts `configure-debug.sh` and `configure-release.sh`
could be used instead to pass reasonable debug and optimization flags 
to g++.

### Implementation details

The YANN library supports sequential, parallel, broadcast, merge, and
arbitrary mapping for layers that enable construction of Neural Networks
with complex topologies. The library tries really hard to avoid memory
allocation and copying buffers, thus providing highly efficient and
fast implementation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.
