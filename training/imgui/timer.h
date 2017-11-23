#pragma once

#ifdef __linux__
#include <cstdint>
#endif

namespace ImGuiControl
{
    class Timer
    {
    public:
        Timer();
        void Reset();
        double GetTime() const;
        float GetMilliseconds() const;

    private:
#if defined(_WIN32)
        double m_start;
        static double s_frequency;
#elif defined(__linux__)
        uint64_t m_start;
#elif defined (__APPLE__)
        unsigned long m_start_sec;
        unsigned long m_start_usec;
#endif
    };
}
