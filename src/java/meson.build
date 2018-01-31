unity_java_src = [
    'com/impossible/glimpse/GlimpseTest.java',
]
android_sdk_jar = join_paths(android_sdk_home, 'platforms/android-23/android.jar')
unity_java_jar = jar('GlimpseTest',
                     unity_java_src,
                     java_args: [
                        '-classpath',  android_sdk_jar,
                     ])

