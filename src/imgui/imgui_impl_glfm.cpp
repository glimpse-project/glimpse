// ImGui GLFM binding with OpenGLES3 + shaders
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID in imgui.cpp.
// (GLFM is a cross-platform general purpose library for handling surfaces, inputs, OpenGL graphics context creation, etc. on mobile devices)
// (epoxy is a helper library to access OpenGL functions since there is no standard header to access modern OpenGL functions easily. Alternatives are GLEW, Glad, etc.)

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(), ImGui::Render() and ImGui_ImplXXXX_Shutdown().
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.
// https://github.com/ocornut/imgui

#include <imgui.h>
#include "imgui_impl_glfm.h"

#include <glfm.h>

#define ARRAY_LEN(X) (sizeof(X)/sizeof(X[0]))

// Data
static double       g_Time = 0.0f;
static bool         g_MouseDown = false;
static ImVec2       g_MousePos(-1, -1);
static bool         g_TouchJustHeld[5] = {};
static bool         g_TouchHeld[5] = {};


static const char* ImGui_ImplGlfm_GetClipboardText(void* user_data)
{
    return NULL;
}

static void ImGui_ImplGlfm_SetClipboardText(void* user_data, const char* text)
{
}

bool ImGui_ImplGlfm_TouchCallback(GLFMDisplay* display, int touch,
                                       GLFMTouchPhase phase, double x, double y)
{
    if (touch > ARRAY_LEN(g_TouchHeld)) {
        return false;
    }

    double scale = glfmGetDisplayScale(display);
    x /= scale;
    y /= scale;

    switch(phase) {
    case GLFMTouchPhaseBegan :
        g_TouchHeld[touch] = g_TouchJustHeld[touch] = true;
        if (touch == 0) {
            g_MousePos = ImVec2(x, y);
        }
        return true;

    case GLFMTouchPhaseHover :
        // Only the emscripten backend of GLFM sends hover (move without touch)
    case GLFMTouchPhaseMoved :
        if (touch == 0) {
            g_MousePos = ImVec2(x, y);
        }
        return true;

    case GLFMTouchPhaseEnded :
    case GLFMTouchPhaseCancelled :
        g_TouchHeld[touch] = false;
        return true;
    }

    return false;
}

bool ImGui_ImplGlfm_KeyCallback(GLFMDisplay *display, GLFMKey keyCode,
                                GLFMKeyAction action, int modifiers)
{
    return false;
}

#if 0
void ImGui_ImplGlfm_MouseButtonCallback(GLFMDisplay*, int button, int action, int /*mods*/)
{
    if (action == GLFW_PRESS && button >= 0 && button < 3)
        g_MouseJustPressed[button] = true;
}

void ImGui_ImplGlmw_ScrollCallback(GLFMDisplay*, double /*xoffset*/, double yoffset)
{
    g_MouseWheel += (float)yoffset; // Use fractional mouse wheel.
}

void ImGui_ImplGlfm_KeyCallback(GLFMDisplay*, int key, int, int action, int mods)
{
    ImGuiIO& io = ImGui::GetIO();
    if (action == GLFW_PRESS)
        io.KeysDown[key] = true;
    if (action == GLFW_RELEASE)
        io.KeysDown[key] = false;

    (void)mods; // Modifiers are not reliable across systems
    io.KeyCtrl = io.KeysDown[GLFW_KEY_LEFT_CONTROL] || io.KeysDown[GLFW_KEY_RIGHT_CONTROL];
    io.KeyShift = io.KeysDown[GLFW_KEY_LEFT_SHIFT] || io.KeysDown[GLFW_KEY_RIGHT_SHIFT];
    io.KeyAlt = io.KeysDown[GLFW_KEY_LEFT_ALT] || io.KeysDown[GLFW_KEY_RIGHT_ALT];
    io.KeySuper = io.KeysDown[GLFW_KEY_LEFT_SUPER] || io.KeysDown[GLFW_KEY_RIGHT_SUPER];
}

void ImGui_ImplGlfm_CharCallback(GLFMDisplay*, unsigned int c)
{
    ImGuiIO& io = ImGui::GetIO();
    if (c > 0 && c < 0x10000)
        io.AddInputCharacter((unsigned short)c);
}
#endif

static void
ImGui_ImplGlfm_UpdateDisplayMetrics(GLFMDisplay* display)
{
    ImGuiIO& io = ImGui::GetIO();

    int w, h;
    glfmGetDisplaySize(display, &w, &h);
    double scale = glfmGetDisplayScale(display);
    io.DisplaySize = ImVec2((float)(w / scale), (float)(h / scale));
    io.DisplayFramebufferScale = ImVec2(scale, scale);
    io.FontGlobalScale = 1.f / scale;
}

bool
ImGui_ImplGlfm_Init(GLFMDisplay* display, bool install_callbacks)
{
    ImGuiIO& io = ImGui::GetIO();

    io.BackendPlatformName = "imgui_impl_glfm";

    io.KeyMap[ImGuiKey_Tab] = GLFMKeyTab;                         // Keyboard mapping. ImGui will use those indices to peek into the io.KeyDown[] array.
    io.KeyMap[ImGuiKey_LeftArrow] = GLFMKeyLeft;
    io.KeyMap[ImGuiKey_RightArrow] = GLFMKeyRight;
    io.KeyMap[ImGuiKey_UpArrow] = GLFMKeyUp;
    io.KeyMap[ImGuiKey_DownArrow] = GLFMKeyDown;
    /*io.KeyMap[ImGuiKey_PageUp] = GLFW_KEY_PAGE_UP;
    io.KeyMap[ImGuiKey_PageDown] = GLFW_KEY_PAGE_DOWN;
    io.KeyMap[ImGuiKey_Home] = GLFW_KEY_HOME;
    io.KeyMap[ImGuiKey_End] = GLFW_KEY_END;
    io.KeyMap[ImGuiKey_Delete] = GLFW_KEY_DELETE;*/
    io.KeyMap[ImGuiKey_Backspace] = GLFMKeyBackspace;
    io.KeyMap[ImGuiKey_Enter] = GLFMKeyEnter;
    io.KeyMap[ImGuiKey_Escape] = GLFMKeyEscape;
    /*io.KeyMap[ImGuiKey_A] = GLFW_KEY_A;
    io.KeyMap[ImGuiKey_C] = GLFW_KEY_C;
    io.KeyMap[ImGuiKey_V] = GLFW_KEY_V;
    io.KeyMap[ImGuiKey_X] = GLFW_KEY_X;
    io.KeyMap[ImGuiKey_Y] = GLFW_KEY_Y;
    io.KeyMap[ImGuiKey_Z] = GLFW_KEY_Z;*/

    io.SetClipboardTextFn = ImGui_ImplGlfm_SetClipboardText;
    io.GetClipboardTextFn = ImGui_ImplGlfm_GetClipboardText;
    io.ClipboardUserData = display;

    ImGui_ImplGlfm_UpdateDisplayMetrics(display);

    if (install_callbacks) {
        glfmSetTouchFunc(display, ImGui_ImplGlfm_TouchCallback);
        glfmSetKeyFunc(display, ImGui_ImplGlfm_KeyCallback);
    }

    return true;
}

void
ImGui_ImplGlfm_Shutdown()
{
}

void
ImGui_ImplGlfm_NewFrame(GLFMDisplay* display, double frametime)
{
    ImGuiIO& io = ImGui::GetIO();
    IM_ASSERT(io.Fonts->IsBuilt() && "Font atlas not built! It is generally built by the renderer back-end. Missing call to renderer _NewFrame() function? e.g. ImGui_ImplOpenGL3_NewFrame().");

    // Setup display size (every frame to accommodate for window resizing)
    ImGui_ImplGlfm_UpdateDisplayMetrics(display);

    // Setup time step
    io.DeltaTime = (float)(frametime - g_Time);
    if (io.DeltaTime < 1.f/60.f) io.DeltaTime = 1.f/60.f;
    g_Time = frametime;

    // Handle fake mouse input
    bool mousedown = false;
    for (int i = 0; i < ARRAY_LEN(g_TouchHeld); ++i) {
        if (g_TouchHeld[i] | g_TouchJustHeld[i]) {
            mousedown = true;
            break;
        }
    }

    if (mousedown && !g_MouseDown) {
        // Delay mouse-down for a frame so that we get a frame with a hover
        // before widget interaction begins.
    } else {
        for (int i = 0; i < ARRAY_LEN(g_TouchHeld); ++i) {
            io.MouseDown[i] = g_TouchHeld[i] || g_TouchJustHeld[i];
            g_TouchJustHeld[i] = false;
        }
    }
    if (io.MousePos.x != g_MousePos.x ||
        io.MousePos.y != g_MousePos.y) {
        io.MousePos = g_MousePos;
    }

    // If the 'mouse' button was just released, tell ImGUI the cursor has
    // disappeared on the next frame.
    // XXX: This breaks combo boxes :(
    /*if (!mousedown && g_MouseDown) {
        g_MousePos = io.MousePos = ImVec2(-FLT_MAX, -FLT_MAX);
    }*/
    g_MouseDown = mousedown;
}
