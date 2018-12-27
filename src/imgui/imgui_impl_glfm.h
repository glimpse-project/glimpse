#pragma once

// ImGui GLFM binding
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID in imgui.cpp.
// (GLFM is a cross-platform general purpose library for handling surfaces, inputs, OpenGL graphics context creation, etc. on mobile devices)
// (epoxy is a helper library to access OpenGL functions since there is no standard header to access modern OpenGL functions easily. Alternatives are GLEW, Glad, etc.)

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(), ImGui::Render() and ImGui_ImplXXXX_Shutdown().
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.
// https://github.com/ocornut/imgui

// dear imgui: Platform Binding for GLFM
// This needs to be used along with a Renderer (e.g. OpenGL3)
// (Info: GLFM is a cross-platform general purpose library for handling windows, inputs, OpenGL ES/ graphics context creation, etc.)

// Implemented features:
//  [ ] Platform: Clipboard support.
//  [ ] Platform: Gamepad support. Enable with 'io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad'.
//  [x] Platform: Mouse cursor shape and visibility.
//  [X] Platform: Keyboard arrays indexed using GLFW_KEY_* codes, e.g. ImGui::IsKeyPressed(GLFW_KEY_SPACE).

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you are new to dear imgui, read examples/README.txt and read the documentation at the top of imgui.cpp.
// https://github.com/ocornut/imgui

struct GLFMDisplay;

#ifdef __cplusplus
extern "C" {
#endif

IMGUI_IMPL_API bool        ImGui_ImplGlfm_Init(GLFMDisplay* display, bool install_callbacks);
IMGUI_IMPL_API void        ImGui_ImplGlfm_Shutdown(GLFMDisplay* display);
IMGUI_IMPL_API void        ImGui_ImplGlfm_NewFrame(GLFMDisplay* display, double frametime);

// GLFM callbacks (installed by default if you enable 'install_callbacks' during initialization)
// Provided here if you want to chain callbacks.
// You can also handle inputs yourself and use those as a reference.
IMGUI_IMPL_API bool        ImGui_ImplGlfm_TouchCallback(GLFMDisplay* display, int touch, int phase, double x, double y);
IMGUI_IMPL_API bool        ImGui_ImplGlfm_KeyCallback(GLFMDisplay *display, int keyCode, int action, int modifiers);

#ifdef __cplusplus
}
#endif

