project('glimpse', ['c', 'cpp'],
        license: 'MIT',
        default_options: ['c_std=gnu11', 'cpp_std=c++14'])

use_system_libs = get_option('use_system_libs')
use_pcl_gbpd = get_option('use_pcl_ground_based_people_detection')

compiler = meson.get_compiler('c')

threads_dep = dependency('threads')
maths_dep = compiler.find_library('m', required : false)

inc = include_directories(
	'src',
	'src/imgui',
	'src/UnityPluginAPI')

client_api_defines = [
    '-DGLM_ENABLE_EXPERIMENTAL'
]

dlib_sp = subproject('dlib')
dlib_dep = dlib_sp.get_variable('dlib_dep')

if use_system_libs
    glm_dep = dependency('glm', fallback: [ 'glm', 'glm_dep'])
    libpng_dep = dependency('libpng', fallback: [ 'libpng', 'png_dep' ])

    pcl_deps = []
    pcl_deps += dependency('pcl_common-1.8', fallback: [ 'pcl', 'common_dep' ])
    pcl_deps += dependency('pcl_features-1.8', fallback: [ 'pcl', 'features_dep' ])
    pcl_deps += dependency('pcl_filters-1.8', fallback: [ 'pcl', 'filters_dep' ])
    pcl_deps += dependency('pcl_kdtree-1.8', fallback: [ 'pcl', 'kdtree_dep' ])
    pcl_deps += dependency('pcl_ml-1.8', fallback: [ 'pcl', 'ml_dep' ])
    pcl_deps += dependency('pcl_octree-1.8', fallback: [ 'pcl', 'octree_dep' ])
    pcl_deps += dependency('pcl_sample_consensus-1.8', fallback: [ 'pcl', 'sample_consensus_dep' ])
    pcl_deps += dependency('pcl_search-1.8', fallback: [ 'pcl', 'search_dep' ])
    pcl_deps += dependency('pcl_segmentation-1.8', fallback: [ 'pcl', 'segmentation_dep' ])

#    pcl_deps += dependency('pcl_io-1.8', fallback: [ 'pcl', 'io_dep' ])
#    pcl_deps += dependency('pcl_keypoints-1.8', fallback: [ 'pcl', 'keypoints_dep' ])
#    pcl_deps += dependency('pcl_recognition-1.8', fallback: [ 'pcl', 'recognition_dep' ])
#    pcl_deps += dependency('pcl_registration-1.8', fallback: [ 'pcl', 'registration_dep' ])
#    pcl_deps += dependency('pcl_stereo-1.8', fallback: [ 'pcl', 'stereo_dep' ])
#    pcl_deps += dependency('pcl_surface-1.8', fallback: [ 'pcl', 'surface_dep' ])
#    pcl_deps += dependency('pcl_tracking-1.8', fallback: [ 'pcl', 'tracking_dep' ])
else
    glm_sp = subproject('glm')
    glm_dep = glm_sp.get_variable('glm_dep')

    png_sp = subproject('libpng')
    libpng_dep = png_sp.get_variable('png_dep')

    pcl_sp = subproject('pcl')
    pcl_deps = []
    pcl_deps += pcl_sp.get_variable('common_dep')
    pcl_deps += pcl_sp.get_variable('segmentation_dep')
endif

freenect_dep = dependency('libfreenect', required: false)
if freenect_dep.found()
    client_api_defines += [ '-DUSE_FREENECT' ]
endif
    
glfw_dep = dependency('glfw3', required: false)
epoxy_dep = dependency('epoxy', required: false)
snappy_dep = dependency('snappy', required: false)

if use_pcl_gbpd
    pcl_deps += dependency('pcl_people-1.8')

    vtk_cpp_args = run_command(['cmake', '--find-package', '-DNAME=VTK', '-DLANGUAGE=CXX', '-DCOMPILER_ID=gcc', '-DMODE=COMPILE']).stdout().strip().split()
    vtk_common_dep = declare_dependency(compile_args: vtk_cpp_args,
                                        link_args: '-lvtkCommonCore')

    client_api_defines += '-DUSE_PCL_GBPD'
else
    vtk_common_dep = []
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
    pcl_deps,
    freenect_dep,
    vtk_common_dep,
]

shared_library('glimpse-unity-plugin',
               [ 'src/glimpse_unity.cc' ] + client_api_src,
               include_directories: inc,
               dependencies: client_api_deps,
               cpp_args: client_api_defines)

if not meson.is_cross_build()

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

executable('train_joint_dist',
           [ 'src/train_joint_dist.cc',
             'src/train_utils.cc',
             'src/image_utils.cc',
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

if snappy_dep.found()
    executable('pack-training-data',
               [ 'src/pack-training-data.cc',
                 'src/pack.c',
                 'src/image_utils.cc',
                 'src/tinyexr.cc',
                 'src/llist.c',
                 'src/xalloc.c' ],
               include_directories: inc,
               dependencies: [ snappy_dep, libpng_dep, threads_dep ])
endif

if glfw_dep.found() and epoxy_dep.found()
    executable('glimpse_viewer',
               [ 'src/glimpse_viewer.cc' ] + client_api_src + imgui_src,
               include_directories: inc,
               dependencies: client_api_deps + [ glfw_dep ],
               cpp_args: client_api_defines)
endif

endif # is_cross_build
