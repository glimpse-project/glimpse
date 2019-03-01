project('glimpse', ['c', 'cpp'],
        license: 'MIT',
        default_options: ['c_std=gnu11', 'cpp_std=c++14'])

use_system_libs = get_option('use_system_libs')
if meson.is_cross_build()
    use_system_libs = false
endif

compiler = meson.get_compiler('c')
python3 = import('python3').find_python()

if compiler.get_define('__ANDROID__') != ''
    add_languages('java')

    android_sdk_home = get_option('android_sdk')
    if android_sdk_home == ''
        android_sdk_home = run_command(python3, '-c',
            'import os; print(os.getenv("ANDROID_HOME", ""))').stdout().strip()
    endif
    if android_sdk_home == ''
        error('Failed to find Android SDK via -Dandroid_sdk option or $ANDROID_HOME environment variable')
    endif

    android_api = '28'
    android_sdk_jar = join_paths(android_sdk_home, 'platforms', 'android-' + android_api, 'android.jar')

    android_build_tools = get_option('android_sdk_build_tools')

    android_ndk_home = get_option('android_ndk')
    if android_ndk_home == ''
        android_ndk_home = run_command(python3, '-c',
            'import os; print(os.getenv("ANDROID_NDK_HOME", ""))').stdout().strip()
    endif
    if android_ndk_home == ''
        error('Failed to find Android NDK via -Dandroid_ndk option or $ANDROID_NDK_HOME environment variable')
    endif

    platform_android = true
else
    platform_android = false
endif

if compiler.get_define('__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__') != ''
    add_languages('objc')

    if get_option('default_library') != 'static'
        error('IOS doesn\'t support shared libraries, so you must configure with --default-library=static')
    endif

    # WARNING WARNING:
    # We have lost/wasted a lot of time investigating issues with UI rotation
    # breaking and for some currently-unfathomable reason it seems to come
    # down to whether the -isysroot points directly to the iPhoneOS.sdk directory
    # or instead points to the same directory via a symlink such as
    # iPhoneOS11.4.sdk -> iPhoneOS.sdk. Things only work as expected if we point
    # at the symlink but that makes it awkward to put in our cross-file if it
    # needs to change regularly and may need to differ depending on how up to
    # date your installation of Xcode is. Using xcrun to query the absolute path
    # of the latest version works around this problem...

    xcrun = find_program('xcrun')
    ios_sdk_path = run_command(xcrun, '--sdk', 'iphoneos', '--show-sdk-path').stdout().strip()
    add_global_arguments([ '-miphoneos-version-min=11.4',
                           '-isysroot', ios_sdk_path ],
                         language: ['c', 'cpp', 'objc'])
    add_global_link_arguments([ '-miphoneos-version-min=11.4',
                                '-isysroot', ios_sdk_path ],
                              language: ['c', 'cpp', 'objc'])

    platform_ios = true
else
    platform_ios = false
endif

platform_mobile = false
platform_pc = false
if platform_ios or platform_android
    platform_mobile = true
else
    platform_pc = true
endif

threads_dep = dependency('threads')
maths_dep = compiler.find_library('m', required : false)


glimpse_includes = [ include_directories('src') ]
if host_machine.system() == 'windows'
glimpse_includes += [ include_directories('src/win32') ]
endif
imgui_includes = [ include_directories('src/imgui') ]
shell_includes = glimpse_includes + imgui_includes
unity_includes = [ include_directories('src/UnityPluginAPI') ]


dlib_sp = subproject('dlib')
dlib_dep = dlib_sp.get_variable('dlib_dep')

shell_defines = []
shell_deps = [ threads_dep ]
shell_src = [
    'src/xalloc.c',
    'src/glimpse_os.c',
    'src/glimpse_mutex.c',
    'src/glimpse_log.c',
    'src/glimpse_assets.c',
    'src/glimpse_imgui_shell.cc',
]
shell_link_args = []

client_api_defines = ['-DGLM_ENABLE_EXPERIMENTAL']
client_api_deps = [ dlib_dep ]

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

    epoxy_dep = dependency('epoxy', fallback: [ 'libepoxy', 'libepoxy_dep' ])
else
    glm_sp = subproject('glm')
    glm_dep = glm_sp.get_variable('glm_dep')

    png_sp = subproject('libpng')
    libpng_dep = png_sp.get_variable('png_dep')

    pcl_sp = subproject('pcl')
    pcl_deps = []
    pcl_deps += pcl_sp.get_variable('common_dep')
    pcl_deps += pcl_sp.get_variable('segmentation_dep')

    epoxy_sp = subproject('libepoxy')
    epoxy_dep = epoxy_sp.get_variable('libepoxy_dep')
endif
client_api_deps += [
    glm_dep,
    libpng_dep,
    pcl_deps,
    epoxy_dep,
]
shell_deps += [
    epoxy_dep,
]

uw_dep = dependency('libunwind', required: false)
if uw_dep.found()
    client_api_defines += ['-DUSE_LIBUNWIND=1']
    client_api_deps += uw_dep
endif

if platform_pc
    freenect_dep = dependency('libfreenect',
                              fallback: ['libfreenect', 'fakenect_dep'],
                              required: false)
    if freenect_dep.found()
        client_api_defines += [ '-DUSE_FREENECT' ]
        client_api_deps += freenect_dep
    endif

    glfw_dep = dependency('glfw3', fallback: [ 'glfw', 'glfw_dep' ], required: false)
    if glfw_dep.found()
        shell_deps += glfw_dep
        shell_defines += [
            '-DUSE_GLFW',
            '-DGLFW_INCLUDE_NONE',
            '-DIMGUI_IMPL_OPENGL_LOADER_CUSTOM=<epoxy/gl.h>'
        ]
        shell_src += [
            'src/imgui/imgui_impl_glfw.cpp',
            'src/imgui/imgui_impl_opengl3.cpp'
        ]
    endif

    snappy_dep = dependency('snappy', required: false)

    rs_dep = dependency('librealsense',
                        fallback: ['librealsense', 'rs_dep'],
                        required: get_option('realsense'))
    if rs_dep.found()
        client_api_defines += [ '-DUSE_REALSENSE' ]
        client_api_deps += rs_dep
    endif
endif

if platform_ios
    glfm_sp = subproject('glfm')
    glfm_dep = glfm_sp.get_variable('glfm_dep')

    shell_deps += glfm_dep
    shell_defines += [
        '-DUSE_GLFM',
        '-DIMGUI_IMPL_OPENGL_LOADER_CUSTOM=<epoxy/gl.h>'
    ]
    shell_src += [
        'src/imgui/imgui_impl_glfm.cpp',
        'src/imgui/imgui_impl_opengl3.cpp',
        'src/ios_utils.m'
    ]
    shell_link_args += [
        '-framework', 'Foundation',
        '-framework', 'OpenGLES',
        '-framework', 'CoreGraphics',
        '-framework', 'QuartzCore',
        '-framework', 'UIKit',
        '-framework', 'AVFoundation',
    ]
endif

if platform_android
    glfm_sp = subproject('glfm')
    glfm_dep = glfm_sp.get_variable('glfm_dep')

    shell_deps += glfm_dep
    shell_defines += [
        '-DUSE_GLFM',
        '-DIMGUI_IMPL_OPENGL_LOADER_CUSTOM=<epoxy/gl.h>'
    ]
    shell_src += [
        'src/imgui/imgui_impl_glfm.cpp',
        'src/imgui/imgui_impl_opengl3.cpp'
    ]
endif


imgui_src = [
    'src/imgui/imgui.cpp',
    'src/imgui/imgui_demo.cpp',
    'src/imgui/imgui_draw.cpp',
    'src/imgui/imgui_widgets.cpp',
    'src/imgui/profiler.cpp',
    'src/imgui/timer.cpp',
]
shell_src += imgui_src


if host_machine.system() == 'windows'
    getopt_src = [ 'src/getopt-compat.c' ]
    getline_src = [ 'src/getline-compat.c' ]
    basename_src = [ 'src/basename-compat.c' ]
else
    getopt_src = []
    getline_src = []
    basename_src = []
endif

client_api_src = [
    'src/glimpse_properties.cc',
    'src/glimpse_context.cc',
    'src/glimpse_device.cc',
    'src/glimpse_record.cc',
    'src/glimpse_assets.c',
    'src/glimpse_data.cc',
    'src/glimpse_mem_pool.cc',
    'src/glimpse_log.c',
    'src/glimpse_gl.c',
    'src/glimpse_target.cc',
    'src/glimpse_mutex.c',
    'src/glimpse_os.c',

    'src/infer_labels.cc',
    'src/joints_inferrer.cc',
    'src/rdt_tree.cc',
    'src/jip.cc',
    'src/xalloc.c',
    'src/image_utils.cc',
    'src/llist.c',

    'src/tinyexr.cc',
    'src/parson.c',
] + basename_src + getline_src


# When building the Unity plugin for iOS we have to explicitly make
# sure to link a monolithic static library instead of declaring
# dependencies that would normally only be resolved when linking
# an executable
if platform_ios
    ios_client_api_libs = []

    # In this case we can assume we didn't find pacakges via pkg-config
    # and so the subproject variables should be valid...
    ios_client_api_libs += png_sp.get_variable('libpng')
    ios_client_api_libs += dlib_sp.get_variable('lib')
    freenect_sp = subproject('libfreenect')
    ios_client_api_libs += freenect_sp.get_variable('fakenect_lib')
    ios_client_api_libs += pcl_sp.get_variable('common_lib')
    ios_client_api_libs += pcl_sp.get_variable('segmentation_lib')
    #ios_client_api_libs += epoxy_sp.get_variable('libepoxy_static')

    client_api_defines += [ '-DUSE_AVF' ]
endif

unity_enabled = false
if get_option('unity_project') != ''
    unity_enabled = true

    if compiler.get_define('__ANDROID__') != ''
        if get_option('unity_editor') == ''
            unity_enabled = false
            warning('')
            warning('')
            warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
            warning('☠                                                                                     ☠')
            warning('☠ -Dunity_editor=<prefix> must be used for build to find .jar class package for Unity ☠')
            warning('☠                                                                                     ☠')
            warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
            warning('')
            warning('')
        endif
        if host_machine.cpu_family() == 'arm64'
            unity_enabled = false
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
    endif

endif

tango_enabled = false
if platform_android
    android_log_dep = declare_dependency(link_args: [ '-llog' ])
    client_api_deps += android_log_dep

    if get_option('android_tango_sdk') != ''
        ndk_arch = meson.get_cross_property('_ndk_arch')
        tango_sdk_inc = join_paths(meson.build_root(), get_option('android_tango_sdk'), 'include')
        tango_sdk_libdir = join_paths(meson.build_root(), get_option('android_tango_sdk'), 'lib', ndk_arch)
        tango_deps = declare_dependency(link_args: [ '-L' + tango_sdk_libdir,
                                                     '-ltango_client_api',
                                                     '-ltango_support_api' ],
                                        compile_args: '-I' + tango_sdk_inc)
        tango_enabled = compiler.has_header('tango_client_api.h',
                                            args: '-I' + tango_sdk_inc)
    endif

    if tango_enabled
        client_api_defines += [ '-DUSE_TANGO' ]
        client_api_deps += [ tango_deps ]
    else
        warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
        warning('☠                                                                                   ☠')
        warning('☠ Tango support not currently enabled. Use -Dandroid_tango_sdk=<path> to locate SDK ☠')
        warning('☠                                                                                   ☠')
        warning('☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠☠')
    endif
endif

need_unity_jar = false
if unity_enabled
    if platform_android
        need_unity_jar = true
    endif

    plugin_deps = client_api_deps
    plugin_defines = client_api_defines

    unity_plugin_src = [ 'src/glimpse_unity.cc' ] + client_api_src
    if platform_ios == true
        unity_plugin_src += [ 'src/ios_utils.m' ]
    endif

    # We want a single monolithic library if linking statically...
    if get_option('default_library') == 'static'
        dep_objects = []
        foreach d: ios_client_api_libs
            dep_objects += d.extract_all_objects()
        endforeach
        plugin_lib = library('glimpse-unity-plugin',
                                     unity_plugin_src,
                                     include_directories: glimpse_includes + unity_includes,
                                     objects: dep_objects,
                                     dependencies: plugin_deps,
                                     c_args: plugin_defines,
                                     cpp_args: plugin_defines)
    else
        plugin_lib = library('glimpse-unity-plugin',
                             unity_plugin_src,
                             include_directories: glimpse_includes + unity_includes,
                             dependencies: plugin_deps,
                             c_args: plugin_defines,
                             cpp_args: plugin_defines)
    endif
endif

need_glimpse_native_jar = false
if platform_android
    viewer_deps = client_api_deps + shell_deps
    viewer_defines = client_api_defines + shell_defines

    if tango_enabled
        viewer_deps += tango_deps
        viewer_defines += [ '-DUSE_TANGO=1' ]
    endif

    use_asset_manager = get_option('use_asset_manager')
    if use_asset_manager
        viewer_defines += [ '-DUSE_ANDROID_ASSET_MANAGER_API' ]
    endif

    android_viewer = shared_library('glimpse_viewer_android',
        [ 'src/glimpse_viewer.cc' ] + client_api_src + shell_src,
        include_directories: glimpse_includes + shell_includes,
        dependencies: viewer_deps,
        c_args: viewer_defines,
        cpp_args: viewer_defines)

    need_glimpse_native_jar = true

    subdir('src/java')
endif

if platform_ios
    viewer_deps = client_api_deps + shell_deps
    viewer_defines = client_api_defines + shell_defines

    executable('glimpse_viewer',
               [ 'src/glimpse_viewer.cc', 'src/ios_utils.m' ] + client_api_src + shell_src,
               include_directories: glimpse_includes + shell_includes,
               dependencies: viewer_deps,
               c_args: viewer_defines,
               cpp_args: viewer_defines,
               link_args: shell_link_args)

    executable('glimpse_shell_test',
               [ 'src/glimpse_shell_test.c', 'src/ios_utils.m' ] + shell_src,
               include_directories: shell_includes,
               dependencies: shell_deps,
               c_args: shell_defines,
               cpp_args: shell_defines,
               link_args: shell_link_args)
endif

if platform_pc

if host_machine.system() != 'windows'  # not ported yet

executable('image-pre-processor',
           [ 'src/image-pre-processor.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/perlin.c',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/glimpse_properties.cc',
             'src/glimpse_data.cc',
             'src/image_utils.cc',
             'src/rdt_tree.cc',
             'src/xalloc.c' ],
           include_directories: glimpse_includes,
           dependencies: [ libpng_dep, threads_dep ])

executable('train_rdt',
           [ 'src/train_rdt.c',
             'src/glimpse_rdt.cc',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/glimpse_properties.cc',
             'src/glimpse_data.cc',
             'src/image_utils.cc',
             'src/rdt_tree.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c'
           ],
           include_directories: glimpse_includes,
           dependencies: [ libpng_dep, threads_dep ])

executable('test_rdt',
           [ 'src/test_rdt.cc',
             'src/glimpse_rdt.cc',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/glimpse_properties.cc',
             'src/glimpse_data.cc',
             'src/image_utils.cc',
             'src/rdt_tree.cc',
             'src/infer_labels.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: glimpse_includes,
           dependencies: [ libpng_dep, threads_dep ])

executable('annotate_bone_map',
           [ 'src/annotate_bone_map.cc',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/glimpse_data.cc',
             'src/image_utils.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: glimpse_includes,
           dependencies: [ libpng_dep, threads_dep, glm_dep ])

executable('train_joint_params',
           [ 'src/train_joint_params.cc',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/glimpse_data.cc',
             'src/infer_labels.cc',
             'src/joints_inferrer.cc',
             'src/image_utils.cc',
             'src/rdt_tree.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c',
             'src/pthread_barrier/pthread_barrier.c' ],
           include_directories: glimpse_includes,
           dependencies: [ libpng_dep, threads_dep ])

executable('depth2labels',
           [ 'src/depth2labels.cc',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/infer_labels.cc',
             'src/image_utils.cc',
             'src/rdt_tree.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: glimpse_includes,
           dependencies: [ libpng_dep, threads_dep ])

executable('recordings-tool',
           [ 'src/recordings-tool.cc' ] + client_api_src,
           c_args: client_api_defines,
           cpp_args: client_api_defines,
           include_directories: glimpse_includes,
           dependencies: client_api_deps)

executable('exr-to-pfm',
           [ 'src/exr-to-pfm.cc',
             'src/tinyexr.cc' ],
           include_directories: glimpse_includes)

executable('pfm-to-exr',
           [ 'src/pfm-to-exr.cc',
             'src/tinyexr.cc' ],
           include_directories: glimpse_includes)

executable('jip-to-json',
           [ 'src/jip-to-json.c',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/jip.cc',
             'src/parson.c',
             'src/llist.c',
             'src/xalloc.c' ],
           include_directories: glimpse_includes,
           dependencies: [ threads_dep ])

executable('pfm-debug',
           [ 'src/pfm-debug.c' ],
           include_directories: glimpse_includes)

executable('index-to-recording',
           [ 'src/index-to-recording.cc',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/glimpse_data.cc',
             'src/image_utils.cc',
             'src/rdt_tree.cc',
             'src/tinyexr.cc',
             'src/parson.c',
             'src/xalloc.c',
           ],
           include_directories: glimpse_includes,
           dependencies: [ libpng_dep, threads_dep ])

if snappy_dep.found() and host_machine.system() == 'linux'
    executable('pack-training-data',
               [ 'src/pack-training-data.cc',
                 'src/pack.c',
                 'src/image_utils.cc',
                 'src/tinyexr.cc',
                 'src/llist.c',
                 'src/xalloc.c' ],
               include_directories: glimpse_includes,
               dependencies: [ snappy_dep, libpng_dep, threads_dep ])
endif

endif # !windows (non-ported tools)

executable('json-to-rdt',
           [ 'src/json-to-rdt.c',
             'src/glimpse_log.c',
             'src/glimpse_mutex.c',
             'src/rdt_tree.cc',
             'src/parson.c',
             'src/xalloc.c' ] + getopt_src,
           include_directories: glimpse_includes,
           dependencies: [ threads_dep ])

if glfw_dep.found() and epoxy_dep.found()
    executable('glimpse_viewer',
               [ 'src/glimpse_viewer.cc' ] + client_api_src + shell_src + getopt_src,
               include_directories: glimpse_includes + shell_includes,
               dependencies: client_api_deps + shell_deps,
               c_args: client_api_defines + shell_defines,
               cpp_args: client_api_defines + shell_defines)

    executable('glimpse_shell_test',
               [ 'src/glimpse_shell_test.c' ] + shell_src + getopt_src,
               include_directories: shell_includes,
               dependencies: shell_deps,
               c_args: shell_defines,
               cpp_args: shell_defines)
endif

endif # platform_pc


if unity_enabled
    install_args = [
        '--buildtype', get_option('buildtype'),
    ]
    install_deps = [ plugin_lib ]

    if compiler.get_define('__ANDROID__') != ''
        plugin_lib_dir = join_paths('Assets', 'Plugins', 'Glimpse', 'Android', 'libs',
                                    meson.get_cross_property('_ndk_arch'))
        install_args += [
            '--android-ndk', android_ndk_home,
            '--android-ndk-arch', meson.get_cross_property('_ndk_arch'),
            '--plugin-jardir', join_paths('Assets', 'Plugins', 'Glimpse', 'Android'),
            '--strip', meson.get_cross_property('_strip_binary'),
        ]
        if tango_enabled
            install_args += [ '--tango-libs', get_option('android_tango_sdk') ]
            install_deps += [ unity_java_jar ]
        endif
        install_deps += [ unity_java_jar ]
    elif platform_ios
        plugin_lib_dir = join_paths('Assets', 'Plugins', 'Glimpse', 'iOS')
    else
        plugin_lib_dir = join_paths('Assets', 'Plugins', 'Glimpse', host_machine.cpu_family())
    endif

    install_args += [
        join_paths(get_option('unity_project')),
        plugin_lib_dir,
    ]
    run_target('install_plugin',
               command: [ 'install_glimpse.py' ] +  install_args,
               depends: install_deps)
endif

if platform_android
    build_apk_args = [
        '--buildtype', get_option('buildtype'),
        '--android-sdk', android_sdk_home,
        '--android-build-tools', android_build_tools,
        '--android-api', android_api,
        '--android-ndk', android_ndk_home,
        '--android-ndk-arch', meson.get_cross_property('_ndk_arch'),
        '--strip', meson.get_cross_property('_strip_binary'),
        'src/glimpse_viewer_android/AndroidManifest.xml',
        'src/glimpse_viewer_android/res',
        'glimpse_viewer',
    ]
    if tango_enabled
        build_apk_args += [ '--tango-libs', get_option('android_tango_sdk') ]
    endif

    run_target('viewer_apk',
               command: [ 'build_apk.py' ] +  build_apk_args,
               depends: [ android_viewer, glimpse_native_jar ])
endif
