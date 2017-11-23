#include "timer.h"

using namespace ImGuiControl;

#if defined(_WIN32)

double Timer::s_frequency = 0.0f;

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <windows.h>

Timer::Timer()
{
	if (s_frequency == 0.0f)
	{
		QueryPerformanceFrequency((LARGE_INTEGER*)&s_frequency);
	}

    QueryPerformanceCounter((LARGE_INTEGER*)&m_start);
}

void Timer::Reset()
{
    QueryPerformanceCounter((LARGE_INTEGER*)&m_start);
}

double Timer::GetTime() const
{
    double counter;
    QueryPerformanceCounter((LARGE_INTEGER*)&counter);
    counter = (counter - m_start) / s_frequency;
    return counter;
}

float Timer::GetMilliseconds() const
{
    double counter;
    QueryPerformanceCounter((LARGE_INTEGER*)&counter);
	float ms = (float)(1000.0f * (counter - m_start) / s_frequency);
    return ms;
}

#elif defined(__linux__)

#include <time.h>
#include <cstdint>

Timer::Timer()
{
    Reset();
}

void Timer::Reset()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    m_start = ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

float Timer::GetMilliseconds() const
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    uint64_t end = ((uint64_t)ts.tv_sec) * 1000000000ULL + (uint64_t)ts.tv_nsec;
    uint64_t duration = end - m_start;

    return duration / 1000000.0f;
}

#elif defined (__APPLE__)

#include <sys/time.h>

Timer::Timer()
{
    Reset();
}

void Timer::Reset()
{
    timeval t;
    gettimeofday(&t, 0);
    m_start_sec = t.tv_sec;
    m_start_usec = t.tv_usec;
}

float Timer::GetMilliseconds() const
{
    timeval t;
    gettimeofday(&t, 0);
    return 1000.0f * (t.tv_sec - m_start_sec) + 0.001f * (t.tv_usec - m_start_usec);
}

#else

Timer::Timer()
{
}

void Timer::Reset()
{
}

float Timer::GetMilliseconds() const
{
	return 0.0f;
}

#endif
