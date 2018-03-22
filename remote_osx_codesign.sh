#!/bin/bash

set -x

#OSX_IP=192.168.1.169
#OSX_USER=new

OSX_IP=192.168.1.127
#OSX_IP=192.168.0.10
OSX_USER=bob

HOST=${OSX_USER}@${OSX_IP}

rm -fr Payload
#cp ../build-ios-debug/Test PayloadUnsigned/Test.app/Test
cp ../build-ios-debug/glimpse_viewer PayloadUnsigned/Test.app/Test
cp -av PayloadUnsigned Payload
rm GlimpseTest.zip
echo "packing zip"
zip -r GlimpseTest.zip Payload
scp GlimpseTest.zip $HOST:

echo "remote..."
ssh $HOST rm -fr Payload
echo "unpacking zip"
ssh $HOST unzip GlimpseTest.zip
echo "signing..."
ssh $HOST <<-EOF
    security default-keychain -s /Users/$OSX_USER/Library/Keychains/glimpse.keychain-db
    security unlock-keychain -p glimpse1234  /Users/$OSX_USER/Library/Keychains/glimpse.keychain-db
    codesign -s "4R8435VF99" --force --entitlements ./Entitlements.plist Payload/Test.app/Test
EOF
echo "packing ipa"
ssh $HOST rm GlimpseTest.ipa
ssh $HOST zip -r GlimpseTest.ipa Payload

echo "retrieving ipa"
scp $HOST:GlimpseTest.ipa .



