#ifndef __AUXILIARY_HPP_
#define __AUXILIARY_HPP_

#include <cmath>
#include <vector>


namespace AuxiliaryNum {
  
    /** @brief Checks if a whole vector<T> is finite. */
    template<typename T>
    inline bool is_finite(const std::vector<T>& vec) {
      
      // Check the whole vector.
      for (size_t i = 0; i < vec.size(); ++i) {
        
        // If either condition satisfies.
        if (std::isinf(vec[i]) || std::isnan(vec[i]))
          
          // Exit: the vector is NOT finite.
          return false;
      }

      // If we get here it means everything is finite.
      return true;
    }
    
    /** @brief Checks if a single value<T> is finite. */
    template<typename T> inline bool is_finite(T value) {
      return (!std::isnan(value) && !std::isinf(value));
    }

    /** @brief Numerical computation of gradient using the Central Difference Formula.

        @param f: the function we want to compute the gradient (Functor).
        @param m: the vector w.r.t. which we want the gradient (i.e. df(m)).

        @note The code is implemented according to:
        @cite[NR] Numerical Recipes in C: The Art of Scientific Computing (2nd Edition),
                  ISBN: 0-521-43108-5. Chapter. 5.7 (Numerical Derivatives).

        @author Michalis Vrettas, PhD.
                E-mail: vrettasm@gmail.com */
    template<typename FTYPE, typename VTYPE>
    std::vector<VTYPE> cdf_grad(FTYPE& f, const std::vector<VTYPE>& m) {
        
      // Sanity check.
      if (m.empty()) {
        throw std::runtime_error("cdf_grad: Parameter vector is empty.");
      }

      // The optimal step when using the CDF should scale
      // with respect to the cubic root of "eps".
      const VTYPE cbrt_eps = cbrt(std::numeric_limits<VTYPE>::epsilon());

      // Return vector.
      std::vector<VTYPE> df(m.size(), VTYPE(0));

      // Make a copy.
      std::vector<VTYPE> x = m;

      // Compute the numerical derivative.
      for (size_t k = 0; k < x.size(); ++k) {
        
        // Step size (for the k-th variable).
        const VTYPE hk = (m[k] != 0.0) ? m[k]*cbrt_eps : cbrt_eps;

        // A small step forwards.
        x[k] = m[k] + hk;

        // Evaluate the function at (x+h).
        const VTYPE f_plus = f(x);

        // A small step backwards.
        x[k] = m[k] - hk;

        // Evaluate the function at (x-h).
        const VTYPE f_minus = f(x);

        // Use central difference formula.
        df[k] = (f_plus - f_minus)/(2.0*hk);

        // Reset the value of the k-th variable.
        x[k] = m[k];
      }

      // Return the numerical gradient.
      return df;
    }
    
}

#endif

// End-of-File.