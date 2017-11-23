#include "profiler.h"
#include "imgui.h"
#include "imgui_internal.h"

using namespace ImGuiControl;

Profiler ImGuiControl::globalInstance;

static inline unsigned int MultiplyColor(unsigned int c, float intensity)
{
    float s = intensity / 255.0f;
    float r = s * (c & 0xFF);
    float g = s * ((c >> 8) & 0xFF);
    float b = s * ((c >> 16) & 0xFF);
    float a = s * (c >> 24);

    unsigned int out;
    out  = ((unsigned int)(((r < 0.0f) ? 0.0f : (r > 1.0f) ? 1.0f : r) * 255.0f + 0.5f));
    out |= ((unsigned int)(((g < 0.0f) ? 0.0f : (g > 1.0f) ? 1.0f : g) * 255.0f + 0.5f)) << 8;
    out |= ((unsigned int)(((b < 0.0f) ? 0.0f : (b > 1.0f) ? 1.0f : b) * 255.0f + 0.5f)) << 16;
    out |= ((unsigned int)(((a < 0.0f) ? 0.0f : (a > 1.0f) ? 1.0f : a) * 255.0f + 0.5f)) << 24;

    return out;
}

static inline int       PositiveModulo(int x, int m)            { return (x % m) < 0 ? (x % m) + m : x % m; }
static inline int       ImSign(int value)                       { return (value < 0 ? -1 : 1); }
static inline int       ImSign(float value)                     { return (value < 0 ? -1 : 1); }
static inline int       ImSign(double value)                    { return (value < 0 ? -1 : 1); }
static inline double   ImMin(double lhs, double rhs)            { return lhs < rhs ? lhs : rhs; }
static inline double   ImMax(double lhs, double rhs)            { return lhs >= rhs ? lhs : rhs; }
static inline double   ImClamp(double v, double mn, double mx)  { return (v < mn) ? mn : (v > mx) ? mx : v; }

void Profiler::Initialize(bool* _isPaused, void (*_setPause)(bool))
{
    m_isPausedPtr = _isPaused;
    m_setPause = _setPause;
    m_timeDuration = 120.0f;
    m_frameAreaMaxDuration = 1000.0f / 30.0f;
    m_frameIndex = -1;
    m_threadIndex = 0;

    m_frameSelectionStart = 2;
    m_frameSelectionEnd = 0;

    memset(m_threads, 0, sizeof(m_threads));

    for (int i = 0; i < MaxThreads; ++i)
    {
        auto& thread = m_threads[i];
        thread.initialized = i == 0;
        thread.sectionIndex = 0;
    }
}

void Profiler::Shutdown()
{
}

void Profiler::NewFrame()
{
    if (IsPaused())
        return;

    float time = m_timer.GetMilliseconds();

    if (m_frameCount > 0)
    {
        auto& previousFrame = m_frames[m_frameIndex];
        previousFrame.endTime = time;
    }

    m_frameIndex = (m_frameIndex + 1) % MaxFrames;
    m_frameCount = ImMin(m_frameCount + 1, MaxFrames);

    auto& frame = m_frames[m_frameIndex];
    frame.index = m_frameIndex;
    frame.startTime = time;
    frame.endTime = -1;
}

void Profiler::InitThreadInternal()
{
    LockCriticalSection2();

    IM_ASSERT(m_threadIndex == -1 && "Thread is already initialized!");

    for (int i = 0; i < MaxThreads; i++)
    {
        auto& thread = m_threads[m_threadIndex];
        if (thread.initialized == false)
        {
            m_threadIndex = i;
            thread.initialized = true;
            thread.activeSectionIndex = -1;
            UnLockCriticalSection2();
            return;
        }
    }

    IM_ASSERT(false && "Every thread slots are used!");
}

void Profiler::FinishThreadInternal()
{
    LockCriticalSection2();
    IM_ASSERT(m_threadIndex > -1 && "Trying to finish an uninitilized thread.");
    auto& thread = m_threads[m_threadIndex];
    thread.initialized = false;
    m_threadIndex = -1;
    UnLockCriticalSection2();
}

void Profiler::PushSectionInternal(const char* name, unsigned int color, const char* fileName, int line)
{
    if (IsPaused())
        return;

    auto& thread = m_threads[m_threadIndex];
    auto& section = thread.sections[thread.sectionIndex];

    if (color == 0x00000000)
    {
        if (thread.callStackDepth > 0 && thread.activeSectionIndex != -1)
        {
            section.color = MultiplyColor(thread.sections[thread.activeSectionIndex].color, 0.8f);
        }
        else
        {
            section.color = Blue;
        }
    }
    else
    {
        section.color = color;
    }

    section.name = name;
    section.fileName = fileName;
    section.line = line;
    section.startTime = m_timer.GetMilliseconds();
    section.endTime = -1;
    section.callStackDepth = thread.callStackDepth;
    section.parentSectionIndex = thread.activeSectionIndex;

    thread.activeSectionIndex = thread.sectionIndex;
    thread.sectionIndex = (thread.sectionIndex + 1) % MaxSections;
    thread.sectionsCount = ImMin(thread.sectionsCount + 1, MaxSections);

    thread.callStackDepth++;
}

void Profiler::PopSectionInternal()
{
    if (IsPaused())
        return;

    auto& thread = m_threads[m_threadIndex];
    thread.callStackDepth--;

    auto& section = thread.sections[thread.activeSectionIndex];
    section.endTime = m_timer.GetMilliseconds();
    thread.activeSectionIndex = section.parentSectionIndex;
}

int ScreenPositionToFrameOffset(float screenPosX, float framesAreaPosMax, float frameWidth, float frameSpacing, int maxFrames)
{
    return ImClamp((int)ceilf((framesAreaPosMax - screenPosX - frameWidth - frameSpacing * 0.5f) / (frameWidth + frameSpacing)), 0, maxFrames - 1);
}

int FrameOffsetToFrameIndex(int currentFrameIndex, int frameIndexOffset, int maxFrames)
{
    return PositiveModulo(currentFrameIndex - frameIndexOffset, maxFrames);
}

struct ThreadInfo
{
    double     minTime;
    double     maxTime;
    int       maxCallStackDepth;
};

void Profiler::DrawUI()
{
    ProfileScopedSection(Profiler UI);

    bool showBorders = false;

    //int normalFontIndex = 0;
    int smallFontIndex = 0;

    // Controls
    float controlsHeight = 15;
    float controlsWidth = 40;
    unsigned int controlsBackColor = 0x88000000;

    // Frames
    float frameAreaHeight = 60;
    unsigned int frameAreaBackColor = 0x60000000;
    unsigned int frameRectColor = 0xAAFFFFFF;
    unsigned int frameNoInfoRectColor = 0x77FFFFFF;
    unsigned int frameSectionWindowColor = 0x55FF8866;
    float frameWidth = 3.0f;
    float frameSpacing = 1.0f;

    // Frame max duration text
    ImVec2 framesMaxDurationTextOffset(-2, 4);
    ImVec2 framesMaxDurationPadding(2, 1);

    // Vertical Slider
    float frameAreaSliderWidth = 0;
    float frameAreaSliderSpacing = 0;
    //unsigned int frameAreaSliderGrabColor = 0x99FFFFFF;
    //unsigned int frameAreaSliderBackColor = 0x22FFFFFF;

    // Timeline
    float timelineHeight = 30;
    unsigned int timlineFrameMarkerColor = 0x22FFFFFF;
    float timelineFrameDurationSpacing = 5;
    unsigned int separatorColor = 0xAA000000;
    
    // Threads    
    float threadTitleWidth = 160;
    float threadTitleHeight = 20.0f;
    float threadSpacing = 10.0f;
    unsigned int threadTitleBackColor = 0x60000000;
    unsigned int threadBackColor = 0x40000000;

    // Sections
    float sectionHeight = 20;
    ImVec2 sectionSpacing = ImVec2(1, 1);
    ImVec2 sectionTextPadding = ImVec2(3, 3);
    float sectionMinWidthForText = 20;
    float sectionWheelZoomSpeed = 0.2f;
    float sectionDragZoomSpeed = 0.01f;
    unsigned int sectionTextColor = 0x90FFFFFF;

    // Interaction
    int selectionButton = 0;
    int panningButton = 0;
    int zoomingButton = 1;

    //-------------------------------------------------------------------------
    // Window
    //-------------------------------------------------------------------------
    char buffer[255];
    auto& style = ImGui::GetStyle();
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);
    ImGui::Begin("Profiler", &m_isWindowOpen);

    //-------------------------------------------------------------------------
    // Controls
    //-------------------------------------------------------------------------
    {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(1, 0));
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[smallFontIndex]);

        ImVec2 controlsAreaSize(ImGui::GetContentRegionAvailWidth(), controlsHeight);
        ImVec2 controlsAreaMin(ImGui::GetCursorScreenPos());
        ImVec2 controlsAreaMax(controlsAreaMin.x + controlsAreaSize.x, controlsAreaMin.y + controlsAreaSize.y);
        ImGui::BeginChild("Controls", controlsAreaSize, showBorders);
        ImGui::GetWindowDrawList()->AddRectFilled(controlsAreaMin, controlsAreaMax, controlsBackColor);

        if (ImGui::Button("Record", ImVec2(controlsWidth, controlsHeight)))
        {
        }

        ImGui::SameLine();
        if (ImGui::Button(IsPaused() ? "Resume" : "Pause", ImVec2(controlsWidth, controlsHeight)))
        {
            SetPause(!IsPaused());
        }

        ImGui::SameLine();
        if (ImGui::Button("Step", ImVec2(controlsWidth, controlsHeight)))
        {
        }

        ImGui::SameLine();
        ImFormatString(buffer, IM_ARRAYSIZE(buffer), "%.1f ms", m_frameAreaMaxDuration);
        if (ImGui::Button(buffer, ImVec2(controlsWidth, controlsHeight)))
        {
            ImGui::OpenPopup("FamesMaxDurationPopup");
        }

        if (ImGui::BeginPopup("FamesMaxDurationPopup"))
        {
            if (ImGui::Selectable("30 FPS")) { m_frameAreaMaxDuration = 1000.0f / 30.f; }
            ImGui::Separator();
            if (ImGui::Selectable("60 FPS")) { m_frameAreaMaxDuration = 1000.0f / 60.f; }
            ImGui::Separator();
            if (ImGui::Selectable("120 FPS")) { m_frameAreaMaxDuration = 1000.0f / 120.0f; }
            ImGui::EndPopup();
        }

        ImGui::EndChild();
        ImGui::PopFont();
        ImGui::PopStyleVar();
    }

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));

    //-------------------------------------------------------------------------
    // Compute min max timings
    //-------------------------------------------------------------------------
    double recordsMinTime = FLT_MAX;
    double recordsMaxTime = -FLT_MAX;
    double minSectionDuration = FLT_MAX;

    ThreadInfo threadInfos[MaxThreads];
    for (int i = 0; i < MaxThreads; i++)
    {
        auto& thread = m_threads[i];
        auto& threadInfo = threadInfos[i];

        threadInfo.minTime = FLT_MAX;
        threadInfo.maxTime = -FLT_MAX;
        threadInfo.maxCallStackDepth = 0;

        if (thread.initialized == false)
            continue;

        for (int j = 0; j < thread.sectionsCount; ++j)
        {
            auto& section = thread.sections[j];
            if (section.endTime < 0)
                continue;

            threadInfo.minTime = ImMin(threadInfo.minTime, section.startTime);
            threadInfo.maxTime = ImMax(threadInfo.maxTime, section.endTime);
            threadInfo.maxCallStackDepth = ImMax(threadInfo.maxCallStackDepth, section.callStackDepth);

            minSectionDuration = ImMin(minSectionDuration, section.endTime - section.startTime);
            recordsMinTime = ImMin(threadInfo.minTime, recordsMinTime);
            recordsMaxTime = ImMax(threadInfo.maxTime, recordsMaxTime);
        }
    }
    double recordsDuration = recordsMaxTime - recordsMinTime;

    //-------------------------------------------------------------------------
    // Move timings if unpaused
    //-------------------------------------------------------------------------
    if (IsPaused() == false)
    {
        auto& startFrame = m_frames[FrameOffsetToFrameIndex(m_frameIndex, m_frameSelectionStart, m_frameCount)];
        auto& endFrame = m_frames[FrameOffsetToFrameIndex(m_frameIndex, m_frameSelectionEnd, m_frameCount)];

        double endTime = endFrame.endTime == -1 ? recordsMaxTime : endFrame.endTime;
        m_timeOffset = recordsMaxTime - endTime;
        m_timeDuration = endTime - startFrame.startTime;
    }

    //-------------------------------------------------------------------------
    // Frames Area
    //-------------------------------------------------------------------------
    ImVec2 framesAreaSize(ImGui::GetContentRegionAvailWidth() - frameAreaSliderWidth - frameAreaSliderSpacing, frameAreaHeight);
    ImGui::BeginChild("Frames", framesAreaSize, showBorders);
    ImVec2 framesAreaMin(ImGui::GetCursorScreenPos());
    ImVec2 framesAreaMax(framesAreaMin.x + framesAreaSize.x, framesAreaMin.y + framesAreaSize.y);

    //-------------------------------------------------------------------------
    // Frames sections selection
    //-------------------------------------------------------------------------
    ImGui::InvisibleButton("##FramesSectionsWindowDummy", framesAreaSize);
    if (ImGui::IsItemActive())
    {
        if (ImGui::IsMouseClicked(selectionButton))
        {
            int frameIndexOffset = ScreenPositionToFrameOffset(ImGui::GetIO().MouseClickedPos[selectionButton].x, framesAreaMax.x, frameWidth, frameSpacing, MaxFrames);
            int frameIndex = FrameOffsetToFrameIndex(m_frameIndex, frameIndexOffset, MaxFrames);
            auto& frame = m_frames[frameIndex];
            if (frame.startTime > 0)
            {
                double endTime = frame.endTime == -1 ? recordsMaxTime : frame.endTime;
                m_timeOffset = recordsMaxTime - endTime;
                m_timeDuration = endTime - frame.startTime;
                m_frameSelectionStart = frameIndexOffset;
                m_frameSelectionEnd = frameIndexOffset;
            }
        }
        else if (ImGui::IsMouseDragging(selectionButton))
        {
            float mousePos = ImGui::GetIO().MouseClickedPos[selectionButton].x;
            int startDragFrameIndexOffset = ScreenPositionToFrameOffset(mousePos, framesAreaMax.x, frameWidth, frameSpacing, MaxFrames);
            int startDragFrameIndex = FrameOffsetToFrameIndex(m_frameIndex, startDragFrameIndexOffset, MaxFrames);
            auto& startDragFrame = m_frames[startDragFrameIndex];

            int endDragFrameIndexOffset = ScreenPositionToFrameOffset(mousePos + ImGui::GetMouseDragDelta(selectionButton).x, framesAreaMax.x, frameWidth, frameSpacing, MaxFrames);
            int endDragFrameIndex = FrameOffsetToFrameIndex(m_frameIndex, endDragFrameIndexOffset, MaxFrames);
            auto& endDragFrame = m_frames[endDragFrameIndex];

            if (startDragFrame.startTime > 0 && endDragFrame.startTime > 0)
            {
                Frame* startFrame;
                Frame* endFrame;

                if (startDragFrame.startTime < endDragFrame.startTime)
                {
                    m_frameSelectionStart = startDragFrameIndexOffset;
                    m_frameSelectionEnd = endDragFrameIndexOffset;
                    startFrame = &startDragFrame;
                    endFrame = &endDragFrame;
                }
                else
                {
                    m_frameSelectionStart = endDragFrameIndexOffset;
                    m_frameSelectionEnd = startDragFrameIndexOffset;
                    startFrame = &endDragFrame;
                    endFrame = &startDragFrame;
                }

                double endTime = endFrame->endTime == -1 ? recordsMaxTime : endFrame->endTime;
                m_timeOffset = recordsMaxTime - endTime;
                m_timeDuration = endTime - startFrame->startTime;

                if (IsPaused() == false)
                {
                    //SetPause(m_frameSelectionEnd != 0);
                }
            }
        }
    }
    ImGui::EndChild();

    //-------------------------------------------------------------------------
    // Vertical slider for fames max duration 
    //-------------------------------------------------------------------------
    //{
    //    ImGui::SameLine();
    //    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + frameAreaSliderSpacing);
    //    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0);
    //    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0);
    //    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImColor(frameAreaSliderBackColor));
    //    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImColor(frameAreaSliderBackColor));
    //    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImColor(frameAreaSliderBackColor));
    //    ImGui::PushStyleColor(ImGuiCol_SliderGrab, ImColor(frameAreaSliderGrabColor));
    //    ImGui::PushStyleColor(ImGuiCol_SliderGrabActive, ImColor(frameAreaSliderGrabColor));
    //    ImGui::VSliderFloat("##frameMaxDuration", ImVec2(frameAreaSliderWidth, frameAreaHeight), &m_frameAreaMaxDuration, 1.0f, 1000.0f / 30.0f, "");
    //    ImGui::PopStyleColor(5);
    //    ImGui::PopStyleVar(2);
    //}

    //-------------------------------------------------------------------------
    // Store interaction button
    //-------------------------------------------------------------------------
    ImVec2 interactionAreaSize(ImGui::GetContentRegionAvailWidth() - threadTitleWidth, ImGui::GetWindowHeight() - ImGui::GetCursorPosY() - 20);
    ImVec2 interactionArea = ImGui::GetCursorScreenPos();
    ImVec2 interactionAreaMin(interactionArea.x + threadTitleWidth, interactionArea.y);
    ImVec2 interactionAreaMax(interactionAreaMin.x + interactionAreaSize.x, interactionAreaMin.y + interactionAreaSize.y);
    //ImGui::GetWindowDrawList()->AddRectFilled(interactionAreaMin, interactionAreaMax, Blue);

    //-------------------------------------------------------------------------
    // Timeline
    //-------------------------------------------------------------------------
    ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX() + threadTitleWidth, ImGui::GetCursorPosY()));
    ImVec2 timelineAreaSize(ImGui::GetContentRegionAvailWidth(), timelineHeight);
    ImGui::BeginChild("Timeline", timelineAreaSize, showBorders, ImGuiWindowFlags_NoMove);
    ImVec2 timelineAreaMin(ImGui::GetCursorScreenPos());
    ImVec2 timelineAreaMax(timelineAreaMin.x + timelineAreaSize.x, timelineAreaMin.y + timelineAreaSize.y);
    ImGui::EndChild();

    //-------------------------------------------------------------------------
    // Threads
    //-------------------------------------------------------------------------
    for (int threadIndex = 0; threadIndex < MaxThreads; ++threadIndex)
    {
        auto& thread = m_threads[threadIndex];
        auto& threadInfo = threadInfos[threadIndex];

        ImGui::PushID(threadIndex);

        ImFormatString(buffer, IM_ARRAYSIZE(buffer), "Thread %d", threadIndex);
        ImGui::SetCursorPos(ImVec2(ImGui::GetCursorPosX(), ImGui::GetCursorPosY() + (threadIndex == 0 ? 0 : threadSpacing)));
        ImVec2 threadStartScreenPos = ImGui::GetCursorScreenPos();

        //---------------------------------------------------------------------
        // Threads Infos
        //---------------------------------------------------------------------
        ImGui::SetCursorPosX(style.WindowPadding.x);
        ImGui::BeginChild(ImGui::GetID("Infos"), ImVec2(threadTitleWidth, threadTitleHeight), showBorders);

        int callStackDepth = 0;
        if (ImGui::TreeNode(buffer, buffer))
        {
            callStackDepth = threadInfo.maxCallStackDepth;
            ImGui::TreePop();
        }
        ImGui::EndChild();

        float threadHeight = (callStackDepth + 1) * (sectionHeight + sectionSpacing.y);

        //---------------------------------------------------------------------
        // Threads Title Background
        //---------------------------------------------------------------------
        ImVec2 backgroundMin = threadStartScreenPos;
        ImVec2 backgroundMax(backgroundMin.x + threadTitleWidth, backgroundMin.y + threadHeight);
        ImGui::GetWindowDrawList()->AddRectFilled(backgroundMin, backgroundMax, threadTitleBackColor);

        //---------------------------------------------------------------------
        // Threads Background
        //---------------------------------------------------------------------
        backgroundMin.x += threadTitleWidth;
        backgroundMin.y += threadTitleWidth;
        backgroundMax.x = backgroundMin.x + ImGui::GetContentRegionAvailWidth() - threadTitleWidth;


        //---------------------------------------------------------------------
        // Sections area
        //---------------------------------------------------------------------
        ImGui::SameLine();
        ImVec2 sectionsAreaSize(ImGui::GetContentRegionAvailWidth(), threadHeight);
        ImGui::BeginChild(ImGui::GetID("Sections"), sectionsAreaSize, showBorders, ImGuiWindowFlags_NoMove);
        ImVec2 sectionsAreaMin(ImGui::GetCursorScreenPos());
        ImVec2 sectionsAreaMax(sectionsAreaMin.x + sectionsAreaSize.x, sectionsAreaMin.y + sectionsAreaSize.y);
        ImGui::GetWindowDrawList()->AddRectFilled(sectionsAreaMin, sectionsAreaMax, threadBackColor);

        //---------------------------------------------------------------------
        // Sections
        //---------------------------------------------------------------------
        for (int sectionIndex = 0; sectionIndex < thread.sectionsCount; ++sectionIndex)
        {
            auto& section = thread.sections[sectionIndex];

            if (section.callStackDepth > callStackDepth)
                continue;

            double endTime = section.endTime;
            if (endTime < 0)
            {
                endTime = threadInfo.maxTime;
            }

            if (endTime < recordsMaxTime - m_timeOffset - m_timeDuration || (section.startTime > recordsMaxTime - m_timeOffset))
                continue;

            float sectionWidth = (float)ImMax(1.0, ((endTime - section.startTime) * sectionsAreaSize.x / m_timeDuration) - sectionSpacing.x);
            float sectionX = (float)(sectionsAreaMin.x + (section.startTime - (recordsMaxTime - m_timeOffset - m_timeDuration)) * (sectionsAreaSize.x / m_timeDuration));
            float sectionY = sectionsAreaMin.y + ((sectionHeight + sectionSpacing.y) * section.callStackDepth);

            ImVec2 rectMin(sectionX, sectionY);
            ImVec2 rectMax(rectMin.x + sectionWidth, rectMin.y + sectionHeight);
            ImVec4 rectClip(rectMin.x, rectMin.y, rectMax.x, rectMax.y);
            ImGui::GetWindowDrawList()->AddRectFilled(rectMin, rectMax, section.color);

            if (sectionWidth > sectionMinWidthForText)
            {
                ImVec2 textPos(rectMin.x + sectionTextPadding.x, rectMin.y + sectionTextPadding.y);
                ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), ImGui::GetFontSize(), textPos, sectionTextColor, section.name, NULL, 0.0f, &rectClip);
            }

            //-----------------------------------------------------------------
            // Tooltip
            //-----------------------------------------------------------------
            if (ImGui::IsMouseHoveringRect(rectMin, rectMax))
            {
                ImGui::SetTooltip("%s (%5.3f ms)\n%s(%d)\n",
                    section.name, endTime - section.startTime,
                    section.fileName, section.line);
            }
        }

        ImGui::EndChild();
        ImGui::PopID();
    }

    //-------------------------------------------------------------------------
    // Interactions on the section area 
    //-------------------------------------------------------------------------
    {
        ImGui::SetCursorScreenPos(interactionAreaMin);
        ImGui::InvisibleButton("", interactionAreaSize);

        //---------------------------------------------------------------------
        // Panning 
        //---------------------------------------------------------------------
        if ((ImGui::IsItemActive() || (ImGui::IsRootWindowOrAnyChildFocused() && ImGui::IsMouseHoveringRect(interactionAreaMin, interactionAreaMax))) && ImGui::IsMouseDragging(panningButton))
        {
            double offset = (ImGui::GetIO().MouseDelta.x * m_timeDuration / interactionAreaSize.x);
            m_timeOffset = ImClamp(m_timeOffset + offset, (double)0, (double)recordsDuration);
            RefreshFrameSelection(recordsMaxTime);
            SetPause(true);
        }

        //-------------------------------------------------------------------------
        // Zooming with mouse drag
        //-------------------------------------------------------------------------
        if (m_sectionAreaDurationWhenZoomStarted == 0 && /*ImGui::IsRootWindowOrAnyChildFocused() && */ImGui::IsMouseHoveringRect(interactionAreaMin, interactionAreaMax) && ImGui::IsMouseDragging(zoomingButton))
        {
            m_sectionAreaDurationWhenZoomStarted = m_timeDuration;
            //SetPause(true);
        }

        if (m_sectionAreaDurationWhenZoomStarted > 0 && ImGui::IsMouseReleased(zoomingButton))
        {
            m_sectionAreaDurationWhenZoomStarted = 0;
        }

        if (m_sectionAreaDurationWhenZoomStarted > 0)
        {
            double oldTime = recordsMaxTime - m_timeOffset;
            double localMousePos = ImGui::GetIO().MouseClickedPos[zoomingButton].x - interactionAreaMin.x;
            double oldTimeOfMouse = (oldTime - m_timeDuration) + (localMousePos * m_timeDuration / interactionAreaSize.x);
            double amount = ImGui::GetMouseDragDelta(zoomingButton).x;
            double zoom = 1 - ImSign(amount) * sectionDragZoomSpeed;
            m_timeDuration = ImClamp(m_sectionAreaDurationWhenZoomStarted * pow(zoom, fabs(amount)), minSectionDuration, recordsDuration);
            double newTimeOfMouse = (oldTime - m_timeDuration) + (localMousePos * m_timeDuration / interactionAreaSize.x);
            double newTime = ImClamp(oldTime + (oldTimeOfMouse - newTimeOfMouse), recordsMinTime, recordsMaxTime);
            m_timeOffset = recordsMaxTime - newTime;
            RefreshFrameSelection(recordsMaxTime);
        }

        //-------------------------------------------------------------------------
        // Zooming with mouse wheel
        //-------------------------------------------------------------------------
        if (/* ImGui::IsRootWindowOrAnyChildFocused() && */ImGui::IsMouseHoveringRect(interactionAreaMin, interactionAreaMax) && ImGui::GetIO().MouseWheel != 0)
        {
            double oldTime = recordsMaxTime - m_timeOffset;
            double localMousePos = ImGui::GetMousePos().x - interactionAreaMin.x;
            double oldTimeOfMouse = (oldTime - m_timeDuration) + (localMousePos * m_timeDuration / interactionAreaSize.x);
            double zoom = 1 - ImSign(ImGui::GetIO().MouseWheel) * sectionWheelZoomSpeed;
            m_timeDuration = ImClamp(m_timeDuration * zoom, minSectionDuration, recordsDuration);
            if (IsPaused())
            {
                double newTimeOfMouse = (oldTime - m_timeDuration) + (localMousePos * m_timeDuration / interactionAreaSize.x);
                double newTime = ImClamp(oldTime + (oldTimeOfMouse - newTimeOfMouse), recordsMinTime, recordsMaxTime);
                m_timeOffset = recordsMaxTime - newTime;
            }
            RefreshFrameSelection(recordsMaxTime);
        }
    }

    //-------------------------------------------------------------------------
    // Separator between thread title and sections
    //-------------------------------------------------------------------------
    ImGui::GetWindowDrawList()->AddLine(timelineAreaMin, ImVec2(timelineAreaMin.x, timelineAreaMin.y + ImGui::GetWindowHeight()), separatorColor, 1.0f);
    ImGui::GetWindowDrawList()->AddLine(ImVec2(0, timelineAreaMin.y), ImVec2(timelineAreaMax.x + ImGui::GetContentRegionAvailWidth(), timelineAreaMin.y), separatorColor, 1.0f);

    //-------------------------------------------------------------------------
    // Frames and timeline rendering
    //-------------------------------------------------------------------------
    {
        ImGui::PushFont(ImGui::GetIO().Fonts->Fonts[smallFontIndex]);

        ImGui::GetWindowDrawList()->AddRectFilled(framesAreaMin, framesAreaMax, frameAreaBackColor);

        for (int i = 0; i < m_frameCount; ++i)
        {
            int frameIndex = PositiveModulo(m_frameIndex - i, m_frameCount);
            auto& frame = m_frames[frameIndex];

            if (frame.startTime <= 0)
                continue;

            double endTime = frame.endTime == -1 ? recordsMaxTime : frame.endTime;
            double frameDuration = endTime - frame.startTime;

            //-----------------------------------------------------------------
            // One frame inside the frame area
            //-----------------------------------------------------------------
            ImVec2 rectMin, rectMax;
            rectMin.x = framesAreaMax.x - i * (frameWidth + frameSpacing) - frameWidth;
            rectMax.x = rectMin.x + frameWidth;

            if (rectMax.x > framesAreaMin.x && rectMin.x < framesAreaMax.x)
            {
                float frameHeight = ImMin((float)(framesAreaSize.y * frameDuration / m_frameAreaMaxDuration), framesAreaSize.y);
                rectMin.y = framesAreaMax.y - frameHeight;
                rectMax.y = framesAreaMax.y;

                unsigned int color = (frame.startTime >= recordsMinTime) ? frameRectColor : frameNoInfoRectColor;
                ImGui::GetWindowDrawList()->AddRectFilled(rectMin, rectMax, color);
            }

            //-----------------------------------------------------------------
            // Frame marker on the timeline and the sections area
            //-----------------------------------------------------------------
            if (frame.startTime >= recordsMaxTime - m_timeOffset - m_timeDuration && (frame.startTime <= recordsMaxTime - m_timeOffset))
            {
                float frameX = (float)(timelineAreaMin.x + (frame.startTime - (recordsMaxTime - m_timeOffset - m_timeDuration)) * (timelineAreaSize.x / m_timeDuration));
                ImGui::GetWindowDrawList()->AddLine(ImVec2(frameX, timelineAreaMin.y), ImVec2(frameX, timelineAreaMin.y + ImGui::GetWindowHeight()), timlineFrameMarkerColor, 1.0f);
            }

            //-----------------------------------------------------------------
            // Frame duration inside the timeline
            //-----------------------------------------------------------------
            if (endTime >= recordsMaxTime - m_timeOffset - m_timeDuration && (endTime <= recordsMaxTime - m_timeOffset))
            {
                ImFormatString(buffer, IM_ARRAYSIZE(buffer), "%5.2f ms", frameDuration);
                ImVec2 textSize = ImGui::CalcTextSize(buffer);

                //float timelineFrameWidth = (float)(timelineAreaSize.x * (endTime - frame.startTime) / m_timeDuration);

                if (textSize.x < frameWidth)
                {
                    ImVec2 textPos;
                    textPos.x = (float)(timelineAreaMin.x - timelineFrameDurationSpacing - textSize.x + (endTime - (recordsMaxTime - m_timeOffset - m_timeDuration)) * (timelineAreaSize.x / m_timeDuration));
                    textPos.y = timelineAreaMin.y + (timelineAreaSize.y - textSize.y) * 0.5f;
                    ImVec4 rectClip(timelineAreaMin.x, timelineAreaMin.y, timelineAreaMax.x, timelineAreaMax.y);
                    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), ImGui::GetFontSize(), textPos, sectionTextColor, buffer, 0, 0, &rectClip);
                }
            }
        }
        ImGui::PopFont();
    }

    //-------------------------------------------------------------------------
    // Frames Sections Window
    //-------------------------------------------------------------------------
    {
        ImVec2 rectMin, rectMax;
        rectMin.x = framesAreaMax.x - ((m_frameSelectionStart + 1) * (frameWidth + frameSpacing)) + frameSpacing;
        rectMin.y = framesAreaMin.y;
        rectMax.x = framesAreaMax.x - ((m_frameSelectionEnd) * (frameWidth + frameSpacing));
        rectMax.y = framesAreaMax.y;

        ImGui::GetWindowDrawList()->AddRectFilled(rectMin, rectMax, frameSectionWindowColor);
    }

    ImGui::PopStyleVar(); // ImGuiStyleVar_ItemSpacing
    ImGui::End();
    ImGui::PopStyleVar(); // ImGuiStyleVar_FrameRounding
    ImGui::PopStyleVar(); // ImGuiStyleVar_WindowPadding
}

void Profiler::RefreshFrameSelection(double recordsMaxTime)
{
    int sectionAreaStartIndex = -1;
    int sectionAreaEndIndex = -1;

    for (int i = 0; i < m_frameCount; ++i)
    {
        int frameIndex = PositiveModulo(m_frameIndex - i, m_frameCount);
        auto& frame = m_frames[frameIndex];

        if (frame.startTime <= 0)
            continue;

        if (sectionAreaStartIndex == -1 && frame.startTime < recordsMaxTime - m_timeOffset - m_timeDuration)
        {
            sectionAreaStartIndex = i;
        }

        if (sectionAreaEndIndex == -1 && frame.startTime < recordsMaxTime - m_timeOffset)
        {
            sectionAreaEndIndex = i;
        }

        if (sectionAreaStartIndex != -1 && sectionAreaEndIndex != -1)
            break;
    }

    m_frameSelectionStart = sectionAreaStartIndex;
    m_frameSelectionEnd = sectionAreaEndIndex;
}