project('glimpse', ['c', 'cpp'],
        default_options: ['c_std=c11', 'cpp_std=c++14'])

use_system_libs = get_option('use_system_libs')

compiler = meson.get_compiler('c')

threads_dep = dependency('threads')
maths_dep = compiler.find_library('m', required : false)

inc = include_directories(
	'training',
        'training/android_stub',
	'training/imgui',
	'training/UnityPluginAPI')

deps = []
if use_system_libs
    deps += dependency('glm')
    deps += dependency('libpng')
    deps += dependency('libfreenect')
    deps += dependency('glfw3')
    deps += dependency('epoxy')
    deps += dependency('dlib-1')

    boost_modules = [
        'date_time',
        'iostreams',
        'filesystem',
        'wserialization',
        'serialization',
        'thread',
        'system'
    ]
    deps += dependency('boost', modules: boost_modules)

    deps += dependency('pcl_common-1.8')
    deps += dependency('pcl_features-1.8')
    deps += dependency('pcl_filters-1.8')
    deps += dependency('pcl_io-1.8')
    deps += dependency('pcl_kdtree-1.8')
    deps += dependency('pcl_keypoints-1.8')
    deps += dependency('pcl_ml-1.8')
    deps += dependency('pcl_octree-1.8')
    deps += dependency('pcl_recognition-1.8')
    deps += dependency('pcl_registration-1.8')
    deps += dependency('pcl_sample_consensus-1.8')
    deps += dependency('pcl_search-1.8')
    deps += dependency('pcl_segmentation-1.8')
    deps += dependency('pcl_stereo-1.8')
    deps += dependency('pcl_surface-1.8')
    deps += dependency('pcl_tracking-1.8')
else
    # TODO 
endif

c_defines = [ '-DGLM_ENABLE_EXPERIMENTAL=1' ]

imgui_src = [
    'training/imgui/imgui.cpp',
    'training/imgui/imgui_demo.cpp',
    'training/imgui/imgui_draw.cpp',
    'training/imgui/profiler.cpp',
    'training/imgui/timer.cpp',
    'training/imgui/imgui_impl_glfw_gles3.cpp',
]

src = [
    'training/glimpse_context.cc',
    'training/glimpse_device.cc',

    'training/infer.cc',
    'training/loader.cc',
    'training/xalloc.c',
    'training/image_utils.cc',
    'training/llist.c',

    'training/tinyexr.cc',
    'training/parson.c',
]

shared_library('glimpse-unity-plugin', src + [ 'training/glimpse_unity.cc' ],
               include_directories: inc,
               dependencies: deps)

executable('glimpse_viewer',
           src + imgui_src + [ 'training/glimpse_viewer.cc' ],
           include_directories: inc,
           dependencies: deps)
