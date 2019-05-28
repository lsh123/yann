# YANN (Yet Another Neural Network library)

YANN is the Neural Network library implemented from ground up using
the [Eigen](https://eigen.tuxfamily.org) library for matrix operations.
The code uses many C++17 features (it was compiled and tested using
g++ 8.3 and should compile on most modern compilers).

### Installing

Install [Eigen](https://eigen.tuxfamily.org) library, for example
on Ubuntu:

```
sudo apt install libeigen3-dev
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
