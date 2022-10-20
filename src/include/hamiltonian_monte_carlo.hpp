#ifndef __HAMILTONIAN_MONTE_CARLO_HPP_
#define __HAMILTONIAN_MONTE_CARLO_HPP_

#include <string>
#include <vector>
#include <boost/filesystem.hpp>
#include "random_number_generator.hpp"

/** @brief Hamiltonian Monte Carlo C++ sampler.

    @author Michalis Vrettas, PhD.
            E-mail: vrettasm@gmail.com */
namespace HamiltonianMC {

  /** @brief Base class that provides the basic interface
      to use HMC for sampling problems.

      @author Michalis Vrettas, PhD.
              E-mail: vrettasm@gmail.com */
  class CalculatesLogPosterior {
    public:
      // Constructor.
      CalculatesLogPosterior() {};

      // Destructor.
      virtual ~CalculatesLogPosterior() {};

      /** Overloaded operator() calls directly the -log(posterior) for
          a given set of model paramerers.
          @param x: the (state) vector where the log(posterior) will
                    be evaluated from the HMC. */
      virtual double operator()(std::vector<double>& x) = 0;
  };

  /** @brief Class which implements the HMC sampler.

      @cite[HMC] Duane S, Kennedy AD, Pendleton BJ, Roweth D. Hybrid Monte Carlo.
                 Physics Letters B. 1987; 195(2):216–222.

      @author Michalis Vrettas, PhD.
              E-mail: vrettasm@gmail.com */
  class HMC {
    public:
      /** @brief Constructor for the HMC.
 
          @param func: Reference to an instance of a class which implements
          the negative log posterior likelihood calculation method.

          @param rng: Reference to a random number generator.
          
          @param saves_path: Directory to save the runtime data. */
      HMC(CalculatesLogPosterior& func, RandomNumberGenerator& rng,
          const std::string& save_path);

      /** @brief Set number of samples. */
      void set_n_samples(const int);

      /** @brief Set the number of burn-in samples. 
          @note These samples will not be included in the final sample. */
      void set_burn_in(const int);

      /** @brief Set the number of leap-frog steps. */
      void set_kappa(const int);

      /** @brief Set the leap-frog step size. */
      void set_dtau(const double);

      /** @brief Set the update frequency parameter. */
      void set_upd_frequency(const int);

      /** @brief Set the scale vector information.
          @note The default scale is 1. However, if we know in advance
          that some parameter will be orders of magnitude different we
          can take this into account, by setting the right scale here. */
      void set_scale(const std::vector<double>&);

      /** @return the number of requested samples. */
      inline int get_n_samples() const { return n_samples; }

      /** @return the number of burn-in samples. */
      inline int get_burn_in() const { return burn_in; }

      /** @return the number of leap frog steps. */
      inline int get_kappa() const { return kappa; }

      /** @return the leap frog step size. */
      inline double get_dtau() const { return dtau; }

      /** @return the frequency of updating the information. */
      inline int get_upd_frequency() const { return upd_frequency; }

      /** @brief Launch HMC sampling procedure from starting position at x0. */
      void run(const std::vector<double>& x0);

      /** @return (a copy) the whole sample. */
      std::vector< std::vector<double> > get_sample() const;

      /** @brief Return vector parameter values from the sample for a specified dimension.
       
          @param idim Index of dimension for which sample values are required.
          
          @return The vector of sample values for dimension idim. */
      std::vector<double> get_par_sample(const int idim) const;

      /** @return a copy of the Energy trace vector. */
      std::vector<double> get_E_trace() const;

      /** @return a copy of the acceptance ratio vector. */
      std::vector<double> get_acc_ratio() const;

      /** @return a copy of the leap-frog step sizes. */
      std::vector<double> get_leap_step_size() const;
      
      /** @brief Saves a record to a txt file.

          It will append the data to the end of the file.
          If the file does not exist it will create it.

          @param rec the record to be saved. */
      void saveRecordToTxt(const std::string& rec) const;
      
      /** @brief Save the HMC data of the sampling procedure in a file.

          @param data vector to be stored.
          @param fname for the data to be stored.
          @param use_comma is used to append comma (or end of line). */
      void saveDataToFile(const std::vector<double>& data,
                          const std::string& fname,
                          const bool use_comma=false) const;

    private:

      /** Total number of samples. */
      int n_samples;

      /** Total no. of burn-in iterations. */
      int burn_in;

      /** Number of leap-frog steps. */
      int kappa;

      /** Step size inside the leap-frog loop. */
      double dtau;

      /** Update frequency for display information. */
      int upd_frequency;
      
      /** Reference for the class that computes the log-posterior. */
      CalculatesLogPosterior& func;

      /** Random number generator. */
      RandomNumberGenerator& rng;
      
      /** Root direcotry for HMC output files. */
      std::string save_path;
      
      /** This is the unique simulation id that is
          used in the files creation. */
      unsigned long sim_ID;
      
      /** Samples from all parameters. */
      std::vector< std::vector<double> > sample;

      /** Energy trace vector. */
      std::vector<double> E_trace;

      /** Acceptance ratio vector. */
      std::vector<double> acc_ratio;

      /** Leap frog steps size. */
      std::vector<double> leap_step;

      /** Scale information of the sampled parameters. */
      std::vector<double> scale;

      /** @note This will prevent accidental copy of the object. */
      HMC(const HMC&) = delete;

      /** @note This will prevent accidental assignment of the object. */
      HMC& operator=(HMC) = delete;
  };
}

#endif

// End-of-File.