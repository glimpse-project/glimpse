#pragma once

// ImGui GLFM binding with OpenGLES 3 + shaders
// In this binding, ImTextureID is used to store an OpenGL 'GLuint' texture identifier. Read the FAQ about ImTextureID in imgui.cpp.
// (GLFM is a cross-platform general purpose library for handling surfaces, inputs, OpenGL graphics context creation, etc. on mobile devices)
// (epoxy is a helper library to access OpenGL functions since there is no standard header to access modern OpenGL functions easily. Alternatives are GLEW, Glad, etc.)

// You can copy and use unmodified imgui_impl_* files in your project. See main.cpp for an example of using this.
// If you use this binding you'll need to call 4 functions: ImGui_ImplXXXX_Init(), ImGui_ImplXXXX_NewFrame(), ImGui::Render() and ImGui_ImplXXXX_Shutdown().
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.
// https://github.com/ocornut/imgui

struct GLFMDisplay;

IMGUI_API bool        ImGui_ImplGlfmGLES3_Init(GLFMDisplay* display, bool install_callbacks);
IMGUI_API void        ImGui_ImplGlfmGLES3_Shutdown(GLFMDisplay* display);
IMGUI_API void        ImGui_ImplGlfmGLES3_NewFrame(GLFMDisplay* display, double frametime);

// Use if you want to reset your rendering device without losing ImGui state.
IMGUI_API void        ImGui_ImplGlfmGLES3_InvalidateDeviceObjects();
IMGUI_API bool        ImGui_ImplGlfmGLES3_CreateDeviceObjects();

// GLFM callbacks (installed by default if you enable 'install_callbacks' during initialization)
// Provided here if you want to chain callbacks.
// You can also handle inputs yourself and use those as a reference.
IMGUI_API bool        ImGui_ImplGlfmGLES3_TouchCallback(GLFMDisplay* display, int touch, int phase, double x, double y);
IMGUI_API bool        ImGui_ImplGlfmGLES3_KeyCallback(GLFMDisplay *display, int keyCode, int action, int modifiers);
