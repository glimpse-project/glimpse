#!/bin/sh

if [ $# != 5 ]; then
  echo "Usage: $0 <build-tools version> <api version> <lib dir> <keystore> <key alias>"
  exit 1
fi

GLIMPSE_ASSETS_ROOT=${GLIMPSE_ASSETS_ROOT:=../}
mkdir -p gen assets
rm -f lib
ln -s $3 lib
#cp $GLIMPSE_ASSETS_ROOT/tree*.rdt $GLIMPSE_ASSETS_ROOT/joint-map.json $GLIMPSE_ASSETS_ROOT/joint-params.json $GLIMPSE_ASSETS_ROOT/joint-dist.json ./assets/

BUILD_TOOLS="$ANDROID_HOME/build-tools/$1"
ANDROID_JAR="$ANDROID_HOME/platforms/android-$2/android.jar"

$BUILD_TOOLS/aapt p -v -f -I $ANDROID_JAR -M AndroidManifest.xml -A assets -S res -m -J gen -F glimpse_viewer.unaligned.apk
$BUILD_TOOLS/aapt a glimpse_viewer.unaligned.apk lib/*/*
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 -keystore $4 glimpse_viewer.unaligned.apk $5
$BUILD_TOOLS/zipalign -f 4 glimpse_viewer.unaligned.apk glimpse_viewer.apk
