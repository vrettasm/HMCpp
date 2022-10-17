#ifndef __SCOPED_LOCK_HPP_
#define __SCOPED_LOCK_HPP_

#include <boost/thread/mutex.hpp>

/** @brief Implements a scoped lock mechanism on a mutex object,
    blocking until the lock is obtained (default), or optionally
    not blocking and recording whether the lock was obtained.

    If block is true (default), instantiation will wait until it
    can lock the mutex it points to, the lock will be released when
    the object goes out of scope.

    If block is false, instantiation will try to lock the mutex,
    but will complete even if this is not possible. If the lock
    was obtained, it will be released when the object goes out of
    scope.

    In all cases calling locked() after instantiation returns true
    if the mutex lock was obtained, false otherwise.

    @author Michalis Vrettas, PhD.
            E-mail: vrettasm@gmail.com */
class ScopedLock {
  public:
    /** @brief Constructor with input parameters.
        @param pm: mutex reference.
        @param block: boolean flag. */
    ScopedLock(boost::mutex& pm, const bool block=true);

    /** @brief Default destructor. */
    ~ScopedLock();

    /** @brief Return the status of the lock. */
    bool locked();
    
  private:
    /** @brief Boolean flag to check if the lock is on/off. */
    bool is_locked;

    /** @brief Mutex that locks the object. */
    boost::mutex& mutex_ref;
};

#endif

// End-of-File.