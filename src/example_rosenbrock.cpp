#include <iostream>
#include <vector>
#include <cmath>

// Custom code.
#include "../src/include/random_number_generator.hpp"
#include "../src/include/hamiltonian_monte_carlo.hpp"

// Compile with:
// g++ -std=c++11 -Wall -g ./common/*.cpp example_multivariate_normal.cpp -o demo02

class Rosenbrock: public HamiltonianMC::CalculatesLogPosterior {
  /* 
   * The Rosenbrock function is a non-convex function which is used
   * as a performance test problem for many optimization algorithms. */
  
  public:
  
    /** @brief Constructor with input parameters. */
    Rosenbrock(double a, double b): alpha(a), beta(b) {};
    
    /** @brief Return the function value f(x, y) 
        for the given parameters alpha, beta. */
    double operator()(std::vector<double>& v) {
      
      // Extract the (x, y) values from the vector.
      double x = v[0];
      double y = v[1];
      
      // Return the f(x, y)
      return std::pow(alpha - x, 2) + beta*std::pow((y - std::pow(x, 2)), 2);
    }
    
  private:
  
    /* Input parameter 'a'. */
    double alpha;

    /* Input parameter 'b'. */
    double beta;
};


int main(int argc, char* argv[]) {
  
  // Create a rosenbrock object
  // with default values.
  Rosenbrock rosen(1.15, 0.5);
  
  // Random number generator.
  RandomNumberGenerator rng;
  
  // Hamiltonian MC object.
  HamiltonianMC::HMC hmc_sampler(rosen, rng,
                                 "/Users/michailvrettas/Desktop/");
  // Set parameters.
  hmc_sampler.set_kappa(50);
  hmc_sampler.set_dtau(0.02);
  hmc_sampler.set_burn_in(500);
  hmc_sampler.set_n_samples(1500);
  
  // Initial search point.
  std::vector<double> x0{0.0, 10.0};
  
  // Start the sampling process.
  hmc_sampler.run(x0);
  
  // Exit.
  return 0;
}
