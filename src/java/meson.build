# XXX: Meson configure_file() doesn't let us specify an output filename
# with a subdirectory so we have to have *another* meson.build file
# next to GlimpseConfig.java.in...
subdir('com/impossible/glimpse')

if need_unity_jar
    unity_java_src = [
        glimpse_config_file,
        'com/impossible/glimpse/GlimpseJNI.java',
        'com/impossible/glimpse/GlimpseUnityActivity.java'
    ]
    unity_jar = join_paths(get_option('unity_editor'),
        'Editor/Data/PlaybackEngines/AndroidPlayer/Variations/mono/Development/Classes/classes.jar')
    unity_java_jar = jar('GlimpseUnity',
                         unity_java_src,

                         # XXX: this is a workaround that results in meson
                         # passing -sourcedir ../src/java (relative to the
                         # build directory) so that the javac -implicit:class
                         # behaviour is able to find the source code of
                         # classes that the current file depends on...
                         include_directories: include_directories('.'),

                         java_args: [
                            '-classpath',  android_sdk_jar + ':' + unity_jar,
                         ])
endif

if need_glimpse_native_jar
    native_java_src = [
        glimpse_config_file,
        'com/impossible/glimpse/GlimpseJNI.java',
        'com/impossible/glimpse/GlimpseNativeActivity.java'
    ]
    glimpse_native_jar = jar('GlimpseNative',
                             native_java_src,

                             # XXX: this is a workaround that results in meson
                             # passing -sourcedir ../src/java (relative to the
                             # build directory) so that the javac -implicit:class
                             # behaviour is able to find the source code of
                             # classes that the current file depends on...
                             include_directories: include_directories('.'),

                             java_args: [
                                '-classpath',  android_sdk_jar,
                             ])
endif
