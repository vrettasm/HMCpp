#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

// Custom code.
#include "../src/include/random_number_generator.hpp"
#include "../src/include/hamiltonian_monte_carlo.hpp"

// Compile with:
// g++ -std=c++11 -Wall -g -I/usr/local/include/eigen3/ ./common/*.cpp example_multivariate_normal.cpp -o demo01

class MultiVariateNormal {
  
  public:
    /** @brief Constructor with input parameters. */
    MultiVariateNormal(const Eigen::VectorXd& mu, const Eigen::MatrixXd& sig):
                       mean(mu), sigma(sig) {};
    
    /* Probability density function. */
    double pdf(const Eigen::VectorXd& x) const {
      
      // Get the number of elements.
      double k = x.size();
      
      // Calculate the sqrt(2*pi).
      const double SQRT2PI = std::sqrt(2.0 * M_PI);
      
      // Get the quadratic form.
      double quadform  = (x - mean).transpose() * sigma.inverse() * (x - mean);
      
      // Calculate the normalization constant.
      double normalization = std::pow(SQRT2PI, -k) *
                             std::pow(sigma.determinant(), -0.5);

      // Return the Gaussian value.
      return normalization * std::exp(-0.5 * quadform);
    }

    /** @brief Return the -log(pdf(x)). */
    double negative_log_pdf(std::vector<double>& x) {
      
      // Create an Eigen::VectorXd from the input 'x'.
      const Eigen::VectorXd vx = Eigen::Map<Eigen::VectorXd,
                                            Eigen::Unaligned>(x.data(), x.size());
      // Return the -log(pdf(x)).
      return -std::log(pdf(vx));
    }
    
  private:
  
    // Mean parameters.
    Eigen::VectorXd mean;

    // Variance parameters.
    Eigen::MatrixXd sigma;
};


class MixtureNormal: public HamiltonianMC::CalculatesLogPosterior {
  
  public:
    // Constructor.
    MixtureNormal(const std::vector<MultiVariateNormal>& vars): mvn(vars){};
    
    // Overloaded operator.
    virtual double operator()(std::vector<double>& x) {
      
      // Create an Eigen::VectorXd from the input 'x'.
      const Eigen::VectorXd vx = Eigen::Map<Eigen::VectorXd,
                                            Eigen::Unaligned>(x.data(), x.size());
      
      // Accumulates the pdf values from each MVN distribution.
      double sum_pdf = 0.0;
      
      // Sum over all the distributions in the vector.
      for (size_t i = 0; i < mvn.size(); ++i) {
        sum_pdf += mvn[i].pdf(vx);
      }
      
      // Return the negative log value of the mixture.
      return (sum_pdf > 0.0) ? -std::log(sum_pdf) : 0.0;
    }
    
  private:
    // Vector with the distributions of the mixture model.
    std::vector<MultiVariateNormal> mvn;
};


int main(int argc, char* argv[]) {
  
  // 1st 2Dim MVN(mu1, sig1):
  Eigen::VectorXd mu1(2);
  mu1 << 0.0, -1.0;
  
  Eigen::MatrixXd sig1(2, 2);
  sig1 << 1.0, 0.1,
          0.1, 1.0;
  
  MultiVariateNormal mvn1(mu1, sig1);
  
  // 2nd 2Dim MVN(mu2, sig2):
  Eigen::VectorXd mu2(2);
  mu2 << -4.0, -6.0;
  
  Eigen::MatrixXd sig2(2, 2);
  sig2 << 0.9, 0.2,
          0.2, 0.9;
  
  MultiVariateNormal mvn2(mu2, sig2);
  
  // 3rd 2Dim MVN(mu3, sig3):
  Eigen::VectorXd mu3(2);
  mu3 << -5.0, 1.0;
  
  Eigen::MatrixXd sig3(2, 2);
  sig3 << 1.2, 0.3,
          0.3, 1.2;
  
  MultiVariateNormal mvn3(mu3, sig3);
  
  // Vector with all the MVN distributions.
  std::vector<MultiVariateNormal> mvn_vector{mvn1, mvn2, mvn3};
  
  // Mixture of Normal Distributions.
  MixtureNormal mixture(mvn_vector);
  
  // Random number generator.
  RandomNumberGenerator rng;
  
  // Hamiltonian MC object.
  HamiltonianMC::HMC hmc_sampler(mixture, rng,
                                 "/Users/michailvrettas/Desktop/");
  // Set parameters.
  hmc_sampler.set_kappa(50);
  hmc_sampler.set_dtau(0.02);
  hmc_sampler.set_burn_in(500);
  hmc_sampler.set_n_samples(1500);
  
  // Initial search point.
  std::vector<double> x0{-4.0, +4.0};
  
  // Start the sampling process.
  hmc_sampler.run(x0);
  
  // Exit.
  return 0;
}
