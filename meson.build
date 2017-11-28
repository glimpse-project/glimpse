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
    glm_dep = dependency('glm')
    libpng_dep = dependency('libpng')
    freenect_dep = dependency('libfreenect')
    glfw_dep = dependency('glfw3')
    epoxy_dep = dependency('epoxy')
    dlib_dep = dependency('dlib-1')

    boost_modules = [
        'date_time',
        'iostreams',
        'filesystem',
        'wserialization',
        'serialization',
        'thread',
        'system'
    ]
    boost_deps = dependency('boost', modules: boost_modules)

    pcl_deps = []
    pcl_deps += dependency('pcl_common-1.8')
    pcl_deps += dependency('pcl_features-1.8')
    pcl_deps += dependency('pcl_filters-1.8')
    pcl_deps += dependency('pcl_io-1.8')
    pcl_deps += dependency('pcl_kdtree-1.8')
    pcl_deps += dependency('pcl_keypoints-1.8')
    pcl_deps += dependency('pcl_ml-1.8')
    pcl_deps += dependency('pcl_octree-1.8')
    pcl_deps += dependency('pcl_recognition-1.8')
    pcl_deps += dependency('pcl_registration-1.8')
    pcl_deps += dependency('pcl_sample_consensus-1.8')
    pcl_deps += dependency('pcl_search-1.8')
    pcl_deps += dependency('pcl_segmentation-1.8')
    pcl_deps += dependency('pcl_stereo-1.8')
    pcl_deps += dependency('pcl_surface-1.8')
    pcl_deps += dependency('pcl_tracking-1.8')
else
    # TODO 
endif

imgui_src = [
    'src/imgui/imgui.cpp',
    'src/imgui/imgui_demo.cpp',
    'src/imgui/imgui_draw.cpp',
    'src/imgui/profiler.cpp',
    'src/imgui/timer.cpp',
    'src/imgui/imgui_impl_glfw_gles3.cpp',
]

client_api_src = [
    'src/glimpse_context.cc',
    'src/glimpse_device.cc',
    'src/glimpse_assets.c',
    'src/glimpse_log.c',

    'src/infer.cc',
    'src/loader.cc',
    'src/xalloc.c',
    'src/image_utils.cc',
    'src/llist.c',

    'src/tinyexr.cc',
    'src/parson.c',
]

client_api_deps = [
    glm_dep,
    libpng_dep,
    epoxy_dep,
    dlib_dep,
    boost_deps,
    pcl_deps,
    freenect_dep
]


executable('image-pre-processor',
           [ 'src/image-pre-processor.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/image_utils.cc',
             'src/xalloc.c' ],
           include_directories: inc,
           dependencies: [ libpng_dep, threads_dep ])

executable('train_rdt',
           [ 'src/train_rdt.cc',
             'src/train_utils.cc',
             'src/image_utils.cc',
             'src/loader.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: inc,
           dependencies: [ libpng_dep, threads_dep ])

executable('train_joint_params',
           [ 'src/train_joint_params.cc',
             'src/infer.cc',
             'src/train_utils.cc',
             'src/image_utils.cc',
             'src/loader.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: inc,
           dependencies: [ libpng_dep, threads_dep ])

executable('depth2labels',
           [ 'src/depth2labels.cc',
             'src/infer.cc',
             'src/image_utils.cc',
             'src/loader.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: inc,
           dependencies: [ libpng_dep, threads_dep ])

executable('exr-to-pfm',
           [ 'src/exr-to-pfm.cc',
             'src/tinyexr.cc' ],
           include_directories: inc)

executable('pfm-to-exr',
           [ 'src/pfm-to-exr.cc',
             'src/tinyexr.cc' ],
           include_directories: inc)

executable('rdt-to-json',
           [ 'src/rdt-to-json.c',
             'src/loader.cc',
             'src/parson.c',
             'src/xalloc.c' ],
           include_directories: inc)

executable('json-to-rdt',
           [ 'src/json-to-rdt.c',
             'src/loader.cc',
             'src/parson.c',
             'src/xalloc.c' ],
           include_directories: inc)

executable('jip-to-json',
           [ 'src/jip-to-json.c',
             'src/loader.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: inc)

executable('pfm-debug',
           [ 'src/pfm-debug.c' ],
           include_directories: inc)

snappy_dep = dependency('snappy')
executable('pack-training-data',
           [ 'src/pack-training-data.cc',
             'src/pack.c',
             'src/image_utils.cc',
             'src/tinyexr.cc',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: inc,
           dependencies: [ snappy_dep, libpng_dep, threads_dep ])

shared_library('glimpse',
               [ 'src/image_utils.cc',
                 'src/infer.cc',
                 'src/loader.cc',
                 'src/tinyexr.cc',
                 'src/parson.c',
                 'src/llist.c',
                 'src/xalloc.c' ],
               include_directories: inc,
               dependencies: [ libpng_dep, threads_dep ])

shared_library('glimpse-unity-plugin',
               [ 'src/glimpse_unity.cc' ] + client_api_src,
               include_directories: inc,
               dependencies: client_api_deps)

executable('glimpse_viewer',
           [ 'src/glimpse_viewer.cc' ] + client_api_src + imgui_src,
           include_directories: inc,
           dependencies: client_api_deps + [ glfw_dep ])
