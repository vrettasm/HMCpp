#ifndef __RANDOM_NUMBER_GENERATOR_HPP_
#define __RANDOM_NUMBER_GENERATOR_HPP_

#include <ctime>
#include <boost/thread/mutex.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "scoped_lock.hpp"

/** @brief Random number generator. Can sample random numbers
    from various distributions. Now threadsafe (using a lock). */

class RandomNumberGenerator {
  
  public:
  
    // Constructor.
    RandomNumberGenerator() {
      seed_from_time_and_pid();
    }

    // Destructor.
    virtual ~RandomNumberGenerator() {};

    /** @brief Seed the generator with the provided value.
        
        @note Thread-safe (uses mutex)
        @note Exception-safe (ScopedLock) */
    void seed(unsigned long _seed) {
      
      // Get the mutex of the object
      // to lock the 'seed' method.
      ScopedLock sl(mutex);
      
      // Seed for the Mersenne Twister.
      mtgen.seed(_seed);
    }

    /** @brief Seed the random number generator from a
        combination of the current time and process id.

        @note The seed may only change once per second. */
    unsigned long seed_from_time_and_pid() {
            
      // Get the process id.
      unsigned long pid = getpid();
      
      // Combine pid with current time.
      unsigned long _seed = time(NULL) + 100*pid;
      
      // Seed Mersenne Twister generator.
      seed(_seed);
      
      // Return the new seed value.
      return _seed;
    }

    /** @brief Fill result specified by result_begin and result_end
        iterators with samples from boost::random distribution dist.

        @note Thread-safe (uses mutex)
        @note Exception-safe (ScopedLock) */
    template<class DISTRIBUTION, typename OutputIterator>
    void sample(DISTRIBUTION& dist, OutputIterator result_begin,
                                    OutputIterator result_end) {
      // Get the mutex of the object
      // to lock the 'sample' method.
      ScopedLock sl(mutex);
      
      // Fill the vector with random numbers from 'dist'.
      for (OutputIterator it = result_begin; it != result_end; ++it) {
        *it = dist(mtgen);
      }
    }

    /** @brief Store a sample from boost::random distribution dist
        in result.

        Thread-safe (uses mutex) */
    template<class DISTRIBUTION, typename SAMPLETYPE>
    void sample(DISTRIBUTION& dist, SAMPLETYPE& result) {
      sample(dist, &result, &result + 1);
    }

    /** @brief Return DTYPE from a uniform random distribution
        in the range [0, 1]. No boost distribution for this.

        Thread-safe (uses mutex) unless threadsafe==false.
        NOT Exception-safe (ScopedLock)!! */
    template <typename DTYPE> DTYPE uniform(const bool threadsafe=true) {
      if (threadsafe) mutex.lock();

      DTYPE result = (mtgen()-mtgen.min())*DTYPE(1)/(mtgen.max()-mtgen.min());

      if (threadsafe) mutex.unlock();

      return result;
    }

    /** @brief Fill result specified by result_begin and result_end
        iterators with samples from a uniform random distribution in
        the range [0, 1].

        @note Thread-safe (uses mutex)
        @note Exception-safe (uses ScopedLock) */
    template<typename DTYPE, typename OutputIterator>
    void uniform(OutputIterator result_begin, OutputIterator result_end) {
      // Get the mutex of the object
      // to lock the 'uniform' method.
      ScopedLock sl(mutex);
      
      // Fill the vector with uniform numbers.
      for (OutputIterator it = result_begin; it != result_end; ++it) {
        *it = uniform<DTYPE>(false);
      }
    }

    /** @brief Fill result with samples from a uniform random
        distribution in the range [0, 1].

        Thread-safe (uses mutex). */
    template<typename DTYPE>
    void uniform(std::vector<DTYPE>& result) {
      uniform<DTYPE>(result.begin(), result.end());
    }
    
    /** @brief Return x sampled from Gaussian probability
        density with N(0, sigma). */
    template <typename DTYPE>
    DTYPE gaussian(const DTYPE sigma) {
      boost::random::normal_distribution<DTYPE> dist(DTYPE(0), sigma);
      DTYPE result = 0;
      sample(dist, result);
      return result;
    }

    /** @brief Fill result vector with samples from a Gaussian
        distribution N(0, Sigma).
         * 
        @note Thread-safe (uses mutex). */
    template<typename DTYPE>
    void gaussian(std::vector<DTYPE>& result, const DTYPE sigma) {
      gaussian<DTYPE>(result.begin(), result.end(), sigma);
    }

    /** @brief Fill a vector with samples from a Gaussian distribution
        N(0, sigma), using iterators result_begin, result_end.
        
        @note Thread-safe (uses mutex). */
    template<typename DTYPE, typename OutputIterator>
    void gaussian(OutputIterator result_begin, OutputIterator result_end,
                  const DTYPE sigma) {
      // Create the Normal distribution object of <DTYPE>.
      boost::random::normal_distribution<DTYPE> dist(DTYPE(0), sigma);
      
      // Use the 'dist' to sample the values.
      sample(dist, result_begin, result_end);
    }

private:
    // Mersenne Twister engine.
    boost::random::mt19937 mtgen;
    
    // Mutex lock.
    mutable boost::mutex mutex;
};

#endif

// End-of-File.