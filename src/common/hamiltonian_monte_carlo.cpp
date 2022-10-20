#include <cmath>
#include <ctime>
#include <chrono>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <iostream>

#include "../include/auxiliary.hpp"
#include "../include/hamiltonian_monte_carlo.hpp"

namespace HamiltonianMC {
  
  HMC::HMC(CalculatesLogPosterior& _func, RandomNumberGenerator& _rng,
           const std::string& _save_path):
           n_samples(10000), burn_in(1000), kappa(100), dtau(1.0e-2),
           upd_frequency(100), func(_func), rng(_rng), save_path(_save_path),
           sim_ID(time(NULL) + getpid()) {};

  void HMC::set_n_samples(const int _n) {
    // Sanity check: positive samples.
    if (_n > 0) {
      n_samples = _n;
    } else {
      
      throw std::invalid_argument(" HMC::set_n_samples:"
                                  " Number of samples can't be negative.");
    }
  }

  void HMC::set_burn_in(const int _n) {
    // Sanity check: positive burn-in.
    if (_n > 0) {
      burn_in = _n;
    } else {
      throw std::invalid_argument(" HMC::set_burn_in:"
                                  " Number of burn-in can't be negative.");
    }
  }

  void HMC::set_kappa(const int _n) {
    // Sanity check: positive leap-frog steps.
    if (_n > 0) {
      kappa = _n;
    } else {
      throw std::invalid_argument(" HMC::set_kappa:"
                                  " Number of leap-frog steps can't be negative.");
    }
  }

  void HMC::set_dtau(const double _n) {
    // Sanity check: positive leap-frog step-size.
    if (_n > 0.0) {
      dtau = _n;
    } else {
      throw std::invalid_argument(" HMC::set_dtau:"
                                  " Leap-frog step size can't be negative.");
    }
  }

  void HMC::set_upd_frequency(const int _n) {
    // Sanity check: positive update frequency.
    if (_n > 0) {
      upd_frequency = _n;
    } else {
      
      throw std::invalid_argument(" HMC::set_upd_frequency:"
                                  " Update frequency can't be negative.");
    }
  }

  void HMC::set_scale(const std::vector<double>& scale0) {
    // Sanity check: make sure we have data.
    if (scale0.empty()) {
      throw std::invalid_argument(" HMC::run:"
                                  " Scaling vector is empty.");
    }
    scale = scale0;
  }

  void HMC::run(const std::vector<double>& x0) {
    
    // Sanity check: make sure we have data.
    if (x0.empty()) {
      throw std::invalid_argument(" HMC::run:"
                                  " Starting position 'x0' is empty.");
    }

    // Length of parameter vector.
    const size_t L = x0.size();

    // Check if scale info exists.
    if (scale.empty()) {
      
      // Default scale is 1.0!
      scale.resize(L, 1.0);
    } else if (scale.size() != L) {
      
      // If there is a missmatch on the sizes something is wrong.
      throw std::invalid_argument(" HMC::run:"
                                  " Sampling vector 'x0' has not the same"
                                  " size as the scale vector.");
    }

    // Define a const NaN of type double.
    const double NaN = std::numeric_limits<double>::quiet_NaN();
    
    // Resize the local storage.
    std::vector< std::vector<double> > store(n_samples, std::vector<double>(L, 0.0));

    // Local copy of the input parameter vector.
    std::vector<double> x_old = x0;

    // Initial negative log posterior (i.e. potential energy).
    double E_old = func(x_old);

    // Initial gradient of 'f()' w.r.t. 'x0'.
    std::vector<double> g_old = AuxiliaryNum::cdf_grad(func, x_old);

    // Acceptance ratio vector.
    acc_ratio.clear();
    acc_ratio.resize(n_samples, NaN);

    // Energy trace vector.
    E_trace.clear();
    E_trace.resize(n_samples, NaN);

    // Leap-frog steps.
    leap_step.clear();
    leap_step.resize(n_samples, NaN);

    // Holds uniformly distributed random variables.
    std::vector<double> uniform_STEP(burn_in + n_samples, NaN);
    std::vector<double> uniform_MHAC(burn_in + n_samples, NaN);
    
    // Fill the vectors ~ U(0,1).
    rng.uniform<double>(uniform_STEP);
    rng.uniform<double>(uniform_MHAC);
    
    // Momentum vector [L x 1].
    std::vector<double> p(L, NaN);
    
    // Accepted samples counter.
    size_t accepted = 0;
    
    // Save the HEADER in the file.
    saveRecordToTxt(" [LOGGING NEW VALUES]");
    
    // Show first message.
    std::cout << "HMC++ started ..." << std::endl;
    
    // Get the first time.
    auto t0 = std::chrono::steady_clock::now();
    
    // Main sampling loop.
    // NOTE: index 'i' -> [burn_in-1 : n_samples]
    //       index 'j' -> [0: nsamples + burn_in]
    for (int i = (1-burn_in), j = 0; i < n_samples; ++i, ++j) {

      // Sample momentum ~ N(mean=0.0, sigma=1.0).
      rng.gaussian<double>(p, 1.0);

      // Kinetic energy (inner-product).
      double E_kin = std::inner_product(p.begin(), p.end(), p.begin(), 0.0);
      
      // Evaluate the Hamiltonian equation.
      double H_old = E_old + 0.5*E_kin;

      // Copy the current states/gradients.
      std::vector<double> x_new = x_old;
      std::vector<double> g_new = g_old;

      // Pick a random direction.
      const double _mu = (uniform_STEP[j] > 0.5) ? +1.0 : -1.0;

      // Length (step-size) in the leapfrog steps (with a small perturbation).
      const double _epsilon = _mu*dtau*(1.0 + 0.1*rng.gaussian<double>(1.0));

      // First half-step of leapfrog.
      for (size_t l = 0; l < L; ++l) {
        // Momentum at 'l-th' position.
        p[l] -= 0.5 * (scale[l]*_epsilon*g_new[l]);

        // State at 'l-th' position.
        x_new[l] += (scale[l]*_epsilon*p[l]);
      }

      // K-th (leapfrog step) gradient.
      std::vector<double> gxk;

      // Random scale factor is a uniform number from [0.8, 1.2].
      const double _scale = 0.8 + 0.4*rng.uniform<double>();
      
      // Number of leap-frog steps is randomized to avoid choosing a trajectory
      // length that happens to produce a near-periodicity for some variable or
      // combination of variables.
      const int _kappa = static_cast<int>(std::max(2.0, std::round(_scale*kappa)));
      
      // Full (kappa-1) leapfrog steps.
      for (int k = 0; k < _kappa-1; ++k) {
        
        // Compute the k-th gradient.
        gxk = AuxiliaryNum::cdf_grad(func, x_new);

        // Compute a full step.
        for (size_t l = 0; l < L; ++l) {
          
          // Momentum at 'l-th' position.
          p[l] -= (scale[l]*_epsilon*gxk[l]);

          // State at 'l-th' position.
          x_new[l] += (scale[l]*_epsilon*p[l]);
        }
      }

      // Compute the energy at the new point.
      double E_new = func(x_new);
      
      // Check if something went wrong.
      if (not AuxiliaryNum::is_finite(E_new)) {
        throw std::runtime_error(" HMC::run: Unexpected error occured! ");
      }

      // Final gradient at 'xnew'.
      g_new = AuxiliaryNum::cdf_grad(func, x_new);

      // Reset E_kin.
      E_kin = 0.0;

      // Final half-step of leapfrog.
      for (size_t l = 0; l < L; ++l) {
        // Momentum at 'l-th' position.
        p[l] -= 0.5*(scale[l]*_epsilon*g_new[l]);

        // Recompute kinetic energy.
        E_kin += p[l]*p[l];
      }

      // Compute the new Hamiltonian.
      double H_new = E_new + 0.5*E_kin;
      
      // Compute the difference between the two Hamiltonians.
      double deltaH = H_old - H_new;

      // Check for acceptance using the Metropolis-Hastings
      // acceptance criterion.
      if (std::min(1.0, std::exp(deltaH)) > uniform_MHAC[j]) {
                
        // Start measuring acceptance after burn-in period.
        if (i >= 0) { accepted++; }
        
        // This happens also during the burn-in.
        g_old = g_new;
        x_old = x_new;
        E_old = E_new;
      }

      // This happens regardless of the acceptance outcome.
      if (i >= 0) {
        
        // Store the current sample.
        store[i] = x_old;

        // Energy value.
        E_trace[i] = E_old;

        // Store the current acceptance ratio.
        acc_ratio[i] = static_cast<double>(accepted)/static_cast<double>(i+1);

        // Leap-frog step size.
        leap_step[i] = _epsilon;

        // Print info every 'upd_frequency' iterations.
        if (i % upd_frequency == 0) {
          
          // New string-stream.
          std::ostringstream oss;
          
          // Construct information record.
          oss << " Itr = " << i << ", Acceptance = " << acc_ratio[i] << ", Energy = " << E_trace[i];
          
          // Save the record in the file.
          saveRecordToTxt(oss.str());
          
          // Print to screen the new record.
          std::cout << oss.str() << std::endl;

        }
        
      }
      
    } // <-- main loop ends here.
        
    // Save the STRING in the file.
    saveRecordToTxt(" [LOGGING STOPPED]\n");

    // Get the final time.
    auto tf = std::chrono::steady_clock::now();
    
    // Save Final message in the runtime file.
    std::cout << "HMC++ finished [" << n_samples << "] in "
              << std::chrono::duration_cast<std::chrono::seconds>(tf - t0).count()
              << " second(s)." << std::endl;
    
    // Save the energy trace.
    saveDataToFile(E_trace, "energy_trace.txt");
    
    // Save the acceptance ratios.
    saveDataToFile(acc_ratio, "acceptance_ratios.txt");

    // Save the leapfrog steps.
    saveDataToFile(leap_step, "leapfrog_steps.txt");

    // Resize the (returned) sample storage.
    sample.resize(L, std::vector<double>(n_samples, NaN));

    // Transpose the sampled values.
    for (size_t l = 0; l < L; ++l) {
      
      for (size_t k = 0; k < n_samples; ++k) {
        sample[l][k] = store[k][l];
      }
      
      // Save the samples of the l-th dimension.
      saveDataToFile(sample[l], "samples.txt", true);
    }

  }

  std::vector< std::vector<double> > HMC::get_sample() const {
    // Check if sample is empty.
    if (sample.empty()) {
      throw std::runtime_error(" HMC::get_sample:"
                               " No sample to return.");
    }
    
    return sample;
  }

  std::vector<double> HMC::get_par_sample(const int idx) const {
    // Check if sample[idx] is empty.
    if (sample[idx].empty()) {
      throw std::runtime_error(" HMC::get_par_sample:"
                               " Sample parameter is empty.");
    }
    
    return sample.at(idx);
  }

  std::vector<double> HMC::get_E_trace() const {
    // Check if the energy trace vector is empty.
    if (E_trace.empty()) {
      throw std::runtime_error(" HMC::get_E_trace:"
                               " Energy trace vector is empty.");
    }
    
    return E_trace;
  }

  std::vector<double> HMC::get_acc_ratio() const {
    // Check if the acceptance ratio vector is empty.
    if (acc_ratio.empty()) {
      throw std::runtime_error(" HMC::get_acc_ratio:"
                               " Acceptance ratio vector is empty.");
    }
    
    return acc_ratio;
  }

  std::vector<double> HMC::get_leap_step_size() const {
    // Check if the vector is empty.
    if (leap_step.empty()) {
      throw std::runtime_error(" HMC::get_leap_step_size:"
                               " Leap-frog step size vector is empty.");
    }
    
    return leap_step;
  }
  
  void HMC::saveRecordToTxt(const std::string& rec_str) const {
    
    // Declare output file stream.
    std::ofstream data_out;
    
    // Create the file name to store the data.
    const boost::filesystem::path file_path(save_path +
                                            std::to_string(sim_ID) +
                                            "_screen_info.txt");
    
    try {
      // Open file (in append mode).
      data_out.open(file_path.string().c_str(), std::ios::app);

      // Create a timestamp.
      std::time_t tstamp = std::time(nullptr);

      // Char array that will hold the formatted date/time.
      char mb_str[40];

      // Add the timestamp in the file.
      if (std::strftime(mb_str, sizeof(mb_str), "%c", std::localtime(&tstamp))) {
        // Add a record.
        data_out << "# " << mb_str << ":" << rec_str << std::endl;
      } else {
        // If the formatting failed for some reason, add the raw timestamp.
        data_out << "# " << tstamp << ":" << rec_str << std::endl;
      }

      // Close file.
      data_out.close();
    } catch (const std::exception& e0) {
      
      // Show what happened.
      std::cout << " HMC::saveRecordToTxt(): " << e0.what() << std::endl;

      // Make sure the file is closed.
      if (data_out.is_open()) {
        data_out.close();
      }
      
    }
  }
  
  void HMC::saveDataToFile(const std::vector<double>& data,
                           const std::string& fname,
                           const bool use_comma) const {
    
    // Declare output file stream.
    std::ofstream data_out;
    
    // Create the file name to store the data.
    const boost::filesystem::path file_path(save_path +
                                            std::to_string(sim_ID) +
                                            "_" + fname);
    try {
      
      // Open file (in append mode).
      data_out.open(file_path.string().c_str(), std::ios::app);
      
      // Copy all the data in the output stream.
      for (size_t i = 0; i < data.size(); ++i) {
        
        // Use a fixed presicion (8 decimals)
        data_out << std::fixed << std::setprecision(8) << data[i];
        
        // If the flag is true we use comma.
        data_out << (use_comma ? "," : "\n");
        
      }
      
      // Final end of line.
      data_out << std::endl;
      
      // Close file.
      data_out.close();
      
    } catch (const std::exception& e0) {
      
      // Show what happened.
      std::cout << " HMC::saveDataToFile(): " << e0.what() << std::endl;

      // Make sure the file is closed.
      if (data_out.is_open()) {
        data_out.close();
      }
      
    }
  }
}
// End-of-File.