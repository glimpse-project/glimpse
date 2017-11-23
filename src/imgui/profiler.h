#pragma once

#include "timer.h"

#define ProfileInitialize(isPaused, commandHandler) ImGuiControl::globalInstance.Initialize(isPaused, commandHandler)
#define ProfileShutdown()                           ImGuiControl::globalInstance.Shutdown()
#define ProfileNewFrame()                           ImGuiControl::globalInstance.NewFrame()
#define ProfileDrawUI()                             ImGuiControl::globalInstance.DrawUI()

#define ProfileInitThread()     ImGuiControl::globalInstance.InitThreadInternal()
#define ProfileFinishThread()   ImGuiControl::globalInstance.FinishThreadInternal()

#define ProfilePushSection_1(x)     ImGuiControl::globalInstance.PushSectionInternal( #x, 0x00000000, __FILE__, __LINE__ )
#define ProfilePushSection_2(x,y)   ImGuiControl::globalInstance.PushSectionInternal( #x, y, __FILE__, __LINE__ )
#define ProfilePushSection_X(x,y,z,...) z

#define ProfilePushSection_FUNC_RECOMPOSER(argsWithParentheses) ProfilePushSection_X argsWithParentheses
#define ProfilePushSection_CHOOSE_FROM_ARG_COUNT(...) ProfilePushSection_FUNC_RECOMPOSER((__VA_ARGS__, ProfilePushSection_2, ProfilePushSection_1, ))
#define ProfilePushSection_MACRO_CHOOSER(...) ProfilePushSection_CHOOSE_FROM_ARG_COUNT(__VA_ARGS__ ())
#define ProfilePushSection(...) ProfilePushSection_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#define ProfilePopSection() gProf.PopSectionInternal()

#define ProfileScopedSection_1(x)   ImGuiControl::ProfileScope S##x__LINE__( #x, 0x00000000, __FILE__, __LINE__ )
#define ProfileScopedSection_2(x,y) ImGuiControl::ProfileScope S##x__LINE__( #x, y, __FILE__, __LINE__ )
#define ProfileScopedSection_X(x,y,z,...) z
#define ProfileScopedSection_FUNC_RECOMPOSER(argsWithParentheses) ProfileScopedSection_X argsWithParentheses
#define ProfileScopedSection_CHOOSE_FROM_ARG_COUNT(...) ProfileScopedSection_FUNC_RECOMPOSER((__VA_ARGS__, ProfileScopedSection_2, ProfileScopedSection_1, ))
#define ProfileScopedSection_MACRO_CHOOSER(...) ProfileScopedSection_CHOOSE_FROM_ARG_COUNT(__VA_ARGS__ ())
#define ProfileScopedSection(...) ProfileScopedSection_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)

#define ProfileFunction0()      ImGuiControl::ProfileScope S##__LINE__( __FUNCTION__, 0x00000000, __FILE__, __LINE__ )
#define ProfileFunction1(x)     ImGuiControl::ProfileScope S##__LINE__( __FUNCTION__, x, __FILE__, __LINE__ )
#define ProfileFunction2(x, y)  ImGuiControl::ProfileScope S##__LINE__( x, y, __FILE__, __LINE__ )


namespace ImGuiControl
{
    class Profiler
    {
    public:
        static const int MaxThreads   = 4;
        static const int MaxFrames    = 600;
        static const int MaxSections  = 2000;

        enum Color : unsigned int
        {
            Yellow      = 0xFF3594B7,
            Red         = 0xFF3945D5,
            Blue        = 0xFFC55344,
            Green       = 0xFF4C8F00,
            Purple      = 0xFF8B377A,
            Dark        = 0xFF222222,
        };

        void Initialize(bool* isPaused, void (*setPause)(bool));
        void Shutdown();
        void NewFrame();
        void InitThreadInternal();
        void FinishThreadInternal();
        void PushSectionInternal(const char* sectionName, unsigned int color, const char* fileName, int line);
        void PopSectionInternal();

        void DrawUI();
        inline void LockCriticalSection2() {}
        inline void UnLockCriticalSection2() {}

        inline bool IsPaused() { return m_isPausedPtr != nullptr && *m_isPausedPtr;  }
        inline void SetPause(bool value) { if (m_setPause != nullptr)  { m_setPause(value); } }

        struct Section
        {
            const char*     name;
            const char*     fileName;
            int             line;
            unsigned int    color;
            int             callStackDepth;
            double          startTime;
            double          endTime;
            int             parentSectionIndex;
        };

        struct Frame
        {
            int         index;
            double      startTime;
            double      endTime;
        };

        struct Thread
        {
            bool        initialized;
            int         callStackDepth;
            int         sectionsCount;
            int         sectionIndex;
            Section     sections[MaxSections];
            int         activeSectionIndex;
        };

        Timer       m_timer;
        Frame       m_frames[MaxFrames];
        int         m_frameCount;
        int         m_frameIndex;
        Thread      m_threads[MaxThreads];
        int         m_threadIndex;
        bool*       m_isPausedPtr;
        void        (*m_setPause)(bool);

        // Ui
        double      m_timeOffset;
        double      m_timeDuration;
        bool        m_isWindowOpen;
        float       m_frameAreaMaxDuration;
        int         m_frameSelectionStart;
        int         m_frameSelectionEnd;
        double      m_sectionAreaDurationWhenZoomStarted;

    private:
        void RefreshFrameSelection(double recordsMaxTime);
    };

    extern Profiler globalInstance;

    struct ProfileScope
    {
        ProfileScope(const char* name, unsigned int color, const char *fileName, int line)
        {
            globalInstance.PushSectionInternal(name, color, fileName, line);
        }

        ~ProfileScope()
        {
            globalInstance.PopSectionInternal();
        }
    };
}

