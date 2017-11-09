#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#include <IUnityInterface.h>
#include <IUnityGraphics.h>

#ifndef ANDROID
#include <epoxy/gl.h>
#endif

static void (*unity_log_function)(const char *msg);

static GLuint video_texture;
static int video_texture_width;
static int video_texture_height;

static float unity_current_time;
static IUnityInterfaces *unity_interfaces;
static IUnityGraphics *unity_graphics;

static UnityGfxRenderer unity_renderer_type = kUnityGfxRendererNull;


extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_notify_log_function(void (*log_func)(const char *msg))
{
    unity_log_function = log_func;
}

static void
unity_log(const char *fmt, ...)
{
    va_list ap;
    char *msg = NULL;

    if (!unity_log_function)
        return;

    va_start(ap, fmt);
    if (vasprintf(&msg, fmt, ap) > 0) {
        unity_log_function(msg);
        free(msg);
    }
    va_end(ap);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_set_time(float time)
{
    unity_current_time = time;
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_set_video_texture(void *handle_as_ptr, int width, int height)
{
    video_texture = *(GLuint *)handle_as_ptr;
    video_texture_width = width;
    video_texture_height = height;
}

static void UNITY_INTERFACE_API
on_graphics_device_event_cb(UnityGfxDeviceEventType type)
{
    switch (type) {
    case kUnityGfxDeviceEventInitialize:
        unity_renderer_type = unity_graphics->GetRenderer();
        break;
    case kUnityGfxDeviceEventShutdown:
        unity_renderer_type = kUnityGfxRendererNull;
        break;
    default:
        break;
    }
}

static void UNITY_INTERFACE_API
on_render_event_cb(int event)
{
    unity_log("Render Event %d DEBUG\n", event);
}

extern "C" UnityRenderingEvent UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
gm_unity_get_render_event_callback(void)
{
    return on_render_event_cb;
}

extern "C" void	UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginLoad(IUnityInterfaces *interfaces)
{
    unity_interfaces = interfaces;
    unity_graphics = interfaces->Get<IUnityGraphics>();
    unity_graphics->RegisterDeviceEventCallback(on_graphics_device_event_cb);

    on_graphics_device_event_cb(kUnityGfxDeviceEventInitialize);
}

extern "C" void UNITY_INTERFACE_EXPORT UNITY_INTERFACE_API
UnityPluginUnload(void)
{
    unity_graphics->UnregisterDeviceEventCallback(on_graphics_device_event_cb);
}
