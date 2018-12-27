
#pragma once

#ifdef _WIN32
#include <windows.h>
typedef CRITICAL_SECTION gm_mutex_t;
#else
#include <pthread.h>
typedef pthread_mutex_t gm_mutex_t;
#endif

/* N.B This is a recursive mutex (since those are the only semantics
 * supported by Windows' CriticalSection or Mutex APIs
 */
void gm_mutex_init(gm_mutex_t *mutex);
void gm_mutex_destroy(gm_mutex_t *mutex);
void gm_mutex_lock(gm_mutex_t *mutex);
void gm_mutex_unlock(gm_mutex_t *mutex);
bool gm_mutex_trylock(gm_mutex_t *mutex);
