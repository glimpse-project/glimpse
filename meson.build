project('glimpse', ['c', 'cpp', 'java'],
        license: 'MIT',
        default_options: ['c_std=gnu11', 'cpp_std=c++14'])

use_system_libs = get_option('use_system_libs')

compiler = meson.get_compiler('c')
python3 = import('python3').find_python()

# Kinda silly how awkward it is to query environment variables...
if compiler.get_define('__ANDROID__') != ''
    android_sdk_home = run_command(python3, '-c',
        'import os; print(os.getenv("ANDROID_HOME", ""))').stdout().strip()
    if android_sdk_home == ''
        error('Failed to find Android SDK via $ANDROID_HOME environment variable')
    endif
endif

introspect_cmd = run_command(python3, '-c',
    'import os; print(os.getenv("MESONINTROSPECT", ""))').stdout().strip()

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

# It's convenient to be able to link with distro packages, but also somewhat
# unpredictable.  We want a way of pinning down our dependencies to subprojects
# when we need to be able to compare results...
if use_system_libs
    glm_dep = dependency('glm', fallback: [ 'glm', 'glm_dep'])
    libpng_dep = dependency('libpng', fallback: [ 'libpng', 'png_dep' ])

    pcl_deps = []

    # XXX: we don't just have multiple dependency() lines with fallbacks
    # because we want to be sure we don't end up resolving one pcl sub-package
    # via pkg-config and another via the 'pcl' subproject.
    #
    # This specific problem arrose for us when building on Ubuntu due to a
    # packaging bug which means many of the pcl_foo-1.8.pc files are dependent
    # on a non-existent pcl_2d-1.8.pc and when Meson runs:
    # `pkg-config --cflags pcl_segmentation-1.8` it fails and falls back to
    # the pcl subproject.
    #
    pcl_common_dep = dependency('pcl_common-1.8', required: false)
    if pcl_common_dep.found()
        pcl_deps += pcl_common_dep
        pcl_dep = dependency('pcl_segmentation-1.8', required: false)
        if not pcl_dep.found()
            error('libpcl pkg-config inconsistency\n' +
                  'If you\'re running Ubuntu then it\'s probably due to these pcl packaging bugs:\n\n' +
                  '  https://bugs.launchpad.net/ubuntu/+source/pcl/+bug/1738902\n\n' +
                  'Workaround:\n\n' +
                  '  $ sudo sed -i \'s/pcl_2d-1.8//\' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_features-1.8.pc' +
                  '  $ sudo sed -i \'s/Requires:/Requires: eigen3 /g\' /usr/lib/x86_64-linux-gnu/pkgconfig/pcl_common-1.8.pc\n\n')
        endif
        pcl_deps += pcl_dep
    else
        pcl_sp = subproject('pcl')
        pcl_deps += pcl_sp.get_variable('common_dep')
        pcl_deps += pcl_sp.get_variable('segmentation_dep')
    endif
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

uw_dep = dependency('libunwind', required: false)

# For now we even use freenect (fakenect) on Android so it's
# not currently optional...
freenect_dep = dependency('libfreenect', required: false)
if not freenect_dep.found()
    freenect_sp = subproject('libfreenect')
    freenect_dep = freenect_sp.get_variable('fakenect_dep')
endif
client_api_defines += [ '-DUSE_FREENECT' ]
    
glfw_dep = dependency('glfw3', required: false)
epoxy_dep = dependency('epoxy', fallback: [ 'libepoxy', 'libepoxy_dep' ])
snappy_dep = dependency('snappy', required: false)

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
    'src/glimpse_mem_pool.cc',
    'src/glimpse_log.c',
    'src/glimpse_gl.c',

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
]

if uw_dep.found()
    client_api_deps += uw_dep
    client_api_defines += '-DUSE_LIBUNWIND=1'
endif

if host_machine.cpu_family() != 'arm64'
    support_unity = true
else
    support_unity = false
    warning('')
    warning('')
    warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
    warning('☠                                                                              ☠')
    warning('☠ Unity doesn\'t support arm64 currently, so not building Glimpse Unity Plugin  ☠')
    warning('☠                                                                              ☠')
    warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
    warning('')
    warning('')
endif

if compiler.get_define('__ANDROID__') != ''
    android_log_dep = declare_dependency(link_args: [ '-llog' ])
    client_api_deps += android_log_dep

    tango_enabled = false

    if get_option('tango_sdk') != ''
        ndk_arch = meson.get_cross_property('_ndk_arch')
        tango_sdk_inc = join_paths(meson.build_root(), get_option('tango_sdk'), 'include')
        tango_sdk_libdir = join_paths(meson.build_root(), get_option('tango_sdk'), 'lib', ndk_arch)
        tango_deps = declare_dependency(link_args: [ '-L' + tango_sdk_libdir,
                                                     '-ltango_support_api' ],
                                        compile_args: '-I' + tango_sdk_inc)
        tango_enabled = compiler.has_header('tango_client_api.h',
                                            args: '-I' + tango_sdk_inc)
    endif

    if tango_enabled
        client_api_deps += tango_deps
        client_api_defines += '-DUSE_TANGO=1'
    else
        warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
        warning('☠                                                                           ☠')
        warning('☠ Tango support not currently enabled. Use -Dtango_sdk=<path> to locate SDK ☠')
        warning('☠                                                                           ☠')
        warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
    endif
endif

if support_unity
    plugin_lib = shared_library('glimpse-unity-plugin',
                                [ 'src/glimpse_unity.cc' ] + client_api_src,
                                include_directories: inc,
                                dependencies: client_api_deps,
                                c_args: client_api_defines,
                                cpp_args: client_api_defines)

    plugin_install_deps = [ plugin_lib ]
    if compiler.get_define('__ANDROID__') != ''
        unity_java_src = [
            'com/impossible/glimpse/GlimpseTest.java'
        ]
        android_sdk_jar = join_paths(android_sdk_home, 'platforms/android-23/android.jar')
        unity_java_jar = jar('GlimpseTest',
                             unity_java_src,
                             java_args: [ '-classpath',  android_sdk_jar ])
        plugin_install_deps += unity_java_jar
    endif
endif

if not meson.is_cross_build()

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
               c_args: client_api_defines,
               cpp_args: client_api_defines)
endif

endif # is_cross_build

if support_unity
    install_args = [
        # XXX: MESONINTROSPECT env not available during run_target() commands
        '--meson-introspect', introspect_cmd,
        '--buildtype', get_option('buildtype'),
        '--unity-project', join_paths(get_option('prefix'))
    ]

    if compiler.get_define('__ANDROID__') != ''
        install_args += [
            '--plugin-libdir', join_paths('Assets/Plugins/Android', 'libs',
                                          meson.get_cross_property('_ndk_arch')),
            '--plugin-jardir', join_paths('Assets/Plugins/Android'),
            '--strip', meson.get_cross_property('_strip_binary'),
            '--android-ndk-arch', meson.get_cross_property('_ndk_arch'),
            '--tango-libs', get_option('android_3rdparty')
        ]
    else
        install_args += [
            '--plugins-subdir', host_machine.cpu_family(),
            '--strip', 'strip'
        ]
    endif

    run_target('install_plugin',
               command: [ 'install_glimpse.py'] +  install_args,
               depends: plugin_install_deps)
endif
