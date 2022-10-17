#include "../include/scoped_lock.hpp"

// Default constructor.
ScopedLock::ScopedLock(boost::mutex& pm, const bool block): mutex_ref(pm) {

  if (block) {
    // Get the lock.
    mutex_ref.lock();
    
    // Update the flag.
    is_locked = true;
  } else{
    
    is_locked = mutex_ref.try_lock();
  }
}

// Default destructor.
ScopedLock::~ScopedLock() {
  
  // Sanity check.
  if (is_locked) {
    
    // Unlock the mutex.
    mutex_ref.unlock();
  }
  
}

// Check if the object is locked.
bool ScopedLock::locked() {
  
  // Return the boolean flag.
  return is_locked;
}

// End-of-File.