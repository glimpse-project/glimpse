project('glimpse', ['c', 'cpp'],
        default_options: ['c_std=gnu11', 'cpp_std=c++14'])

use_system_libs = get_option('use_system_libs')

compiler = meson.get_compiler('c')

threads_dep = dependency('threads')
maths_dep = compiler.find_library('m', required : false)

inc = include_directories(
	'src',
        'src/android_stub',
	'src/imgui',
	'src/UnityPluginAPI')

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
    'src/imgui/imgui.cpp',
    'src/imgui/imgui_demo.cpp',
    'src/imgui/imgui_draw.cpp',
    'src/imgui/profiler.cpp',
    'src/imgui/timer.cpp',
    'src/imgui/imgui_impl_glfw_gles3.cpp',
]

src = [
    'src/glimpse_context.cc',
    'src/glimpse_device.cc',
    'src/glimpse_assets.c',

    'src/infer.cc',
    'src/loader.cc',
    'src/xalloc.c',
    'src/image_utils.cc',
    'src/llist.c',

    'src/tinyexr.cc',
    'src/parson.c',
]

shared_library('glimpse-unity-plugin', src + [ 'src/glimpse_unity.cc' ],
               include_directories: inc,
               dependencies: deps)

executable('glimpse_viewer',
           src + imgui_src + [ 'src/glimpse_viewer.cc' ],
           include_directories: inc,
           dependencies: deps)
