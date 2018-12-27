#!/bin/bash

# TODO: remove me - this script was just used to figure out how to cross-compile
# for windows using Clang

set -x

VC_INCLUDES="/mnt/c/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.15.26726/include"
VC_LIBS="/mnt/c/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Tools/MSVC/14.15.26726/lib/x64"

# Universal C Runtime Library (UCRT):
CRT_INCLUDES="/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.17134.0/ucrt"
CRT_LIBS="/mnt/c/Program Files (x86)/Windows Kits/10/Lib/10.0.17134.0/ucrt/x64"

# User-Mode Library (UM):
UM_INCLUDES="/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.17134.0/um"
UM_LIBS="/mnt/c/Program Files (x86)/Windows Kits/10/Lib/10.0.17134.0/um/x64"


# Comment out to directly use headers/include from Windows
VC_INCLUDES=../windows-sdk/vc/include
VC_LIBS=../windows-sdk/vc/lib/x64
CRT_INCLUDES=../windows-sdk/ucrt/include
CRT_LIBS=../windows-sdk/ucrt/lib/x64
UM_INCLUDES=../windows-sdk/sdk/um/include
UM_LIBS=../windows-sdk/sdk/um/lib/x64

clang-8 \
	-target x86_64-pc-windows \
	-fms-extensions \
	-fms-compatibility \
	-fdelayed-template-parsing \
        -Wno-expansion-to-defined \
        -Wno-ignored-attributes \
	-isystem "$VC_INCLUDES" \
	-isystem "$CRT_INCLUDES" \
	-isystem "$UM_INCLUDES" \
        -D_WINSOCK_DEPRECATED_NO_WARNINGS=1 \
        -D_CRT_NONSTDC_NO_DEPRECATE \
        -D_CRT_SECURE_NO_DEPRECATE \
        -D_CRT_NONSTDC_NO_WARNINGS \
        -D_CRT_SECURE_NO_WARNINGS \
        -march=native \
        -mtune=native \
	-fuse-ld=lld \
        -DIS_DLL \
        -g \
        -c \
        -o hello.obj \
        -v \
        hello.c

clang-8 \
	-target x86_64-pc-windows \
	-fms-extensions \
	-fms-compatibility \
	-fdelayed-template-parsing \
	"-L$VC_LIBS" \
	"-L$CRT_LIBS" \
	"-L$UM_LIBS" \
        -march=native \
        -mtune=native \
	-fuse-ld=lld \
        -nostdlib \
        -nostdlib++ \
        -lmsvcrtd \
        -lvcruntimed \
        -lucrtd \
        -lopengl32 \
        -Wl,/implib:hello.lib \
        -shared \
        -o hello.dll \
        hello.obj

clang-8 \
	-target x86_64-pc-windows \
	-fms-extensions \
	-fms-compatibility \
        -Wno-expansion-to-defined \
        -Wno-ignored-attributes \
	-isystem "$VC_INCLUDES" "-L$VC_LIBS" \
	-isystem "$CRT_INCLUDES" "-L$CRT_LIBS" \
	-isystem "$UM_INCLUDES" "-L$UM_LIBS" \
        -D_WINSOCK_DEPRECATED_NO_WARNINGS=1 \
        -D_CRT_NONSTDC_NO_DEPRECATE \
        -D_CRT_SECURE_NO_DEPRECATE \
        -D_CRT_NONSTDC_NO_WARNINGS \
        -D_CRT_SECURE_NO_WARNINGS \
        -march=native \
        -mtune=native \
	-fuse-ld=lld \
        -nostdlib \
        -nostdlib++ \
        -lmsvcrtd \
        -lvcruntimed \
        -lucrtd \
        -lopengl32 \
        -lhello \
        -o hello.exe \
        -g \
        -v \
        hello.c

