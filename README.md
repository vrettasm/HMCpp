# Hamiltonian Monte Carlo (HMC)

Hamiltonian Monte Carlo (HMC) sampling method in C++11.

### References

The original paper, that introduced this method is described in:

   1. Simon Duane, Anthony D. Kennedy, Brian J. Pendleton and Duncan Roweth (1987).
   "Hybrid Monte Carlo". Physics Letters B. 195 (2): 216–222.

Several implementation details are given in:

   2. Radford M. Neal (1996). "Monte Carlo Implementation".
   Bayesian Learning for Neural Networks. Springer. pp. 55–98.


### Requirements

   > Boost library is required

### Examples

The examples have been compiled (successfully) on OSX10.14 with:

  > g++ --version
  >
  > Apple LLVM version 10.0.1 (clang-1001.0.46.4)
  > Target: x86_64-apple-darwin18.7.0
  >

1. [Rosenbrock](examples/example_rosenbrock.cpp), compile with:

  > g++ -std=c++11 -Wall -g ../src/common/*.cpp example_rosenbrock.cpp -o demo01

2. [Multivariate Normal](examples/example_multivariate_normal.cpp) This example
  uses "Eigen/Dense" to perform the matrix/vector operations of the pdfs. This
  library is need ONLY for the example to run NOT for the HMC method.

  Compile with:

  > g++ -std=c++11 -Wall -g -I/usr/local/include/eigen3/ ../src/common/*.cpp
    example_multivariate_normal.cpp -o demo02

Upon completion the algorithm will output the results in text files for further
processing.

### Unittests

   Coming soon
