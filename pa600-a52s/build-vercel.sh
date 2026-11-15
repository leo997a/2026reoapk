#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
export ANDROID_HOME="$PWD/.android-sdk"
export ANDROID_SDK_ROOT="$ANDROID_HOME"
export GRADLE_USER_HOME="$PWD/.gradle-user"
mkdir -p "$ANDROID_HOME/cmdline-tools" "$GRADLE_USER_HOME" "$PWD/dist" "$PWD/.tools"

echo "Checking Java toolchain"
if ! java -version 2>&1 | head -1 | grep -Eq 'version "(17|18|19|2[0-9])\.'; then
  echo "Installing portable JDK 17"
  curl -fsSL --retry 3 -o .tools/jdk.tar.gz "https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse"
  mkdir -p .tools/jdk
  tar -xzf .tools/jdk.tar.gz -C .tools/jdk --strip-components=1
  export JAVA_HOME="$PWD/.tools/jdk"
  export PATH="$JAVA_HOME/bin:$PATH"
fi
java -version

echo "Installing Android SDK commandline tools"
curl -fsSL --retry 3 -o .tools/cmdline.zip "https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
unzip -q .tools/cmdline.zip -d .tools/cmdline-unpacked
mv .tools/cmdline-unpacked/cmdline-tools "$ANDROID_HOME/cmdline-tools/latest"
export PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH"
yes | sdkmanager --sdk_root="$ANDROID_HOME" --licenses >/dev/null || true
sdkmanager --sdk_root="$ANDROID_HOME" "platforms;android-35" "build-tools;35.0.0"

echo "Downloading Gradle distribution"
curl -fsSL --retry 3 -o .tools/gradle.zip "https://services.gradle.org/distributions/gradle-8.9-bin.zip"
unzip -q .tools/gradle.zip -d .tools
export PATH="$PWD/.tools/gradle-8.9/bin:$PATH"
echo "Building Android APK"
gradle --no-daemon :app:assembleDebug
cp app/build/outputs/apk/debug/app-debug.apk dist/REO-Pa600-A52s.apk
sha256sum dist/REO-Pa600-A52s.apk | tee dist/SHA256.txt
echo "APK build succeeded"
