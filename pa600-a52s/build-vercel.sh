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
cp app/build/outputs/apk/debug/app-debug.apk dist/REO-Pa600QT-v4-A52s.apk
sha256sum dist/REO-Pa600QT-v4-A52s.apk | tee dist/SHA256.txt

cat > dist/index.html <<'HTML'
<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>REO Pa600QT v4 — Galaxy A52s</title><style>*{box-sizing:border-box}body{margin:0;background:#101827;color:#f5f6f2;font:16px/1.8 system-ui,-apple-system,"Segoe UI",sans-serif;display:flex;min-height:100svh;align-items:center;justify-content:center;padding:22px}main{background:#1b2938;max-width:540px;border-radius:23px;padding:32px;box-shadow:0 12px 45px #0005}h1{line-height:1.25;margin:5px 0 18px}p{color:#c3d2d7}a{display:block;text-align:center;background:#69baa8;color:#10231e;border-radius:12px;padding:14px;text-decoration:none;font-weight:800;font-size:18px;margin:23px 0}.small{font-size:13px;color:#9cabb1;overflow-wrap:anywhere}code{direction:ltr;display:block;text-align:left;font-size:11px;word-break:break-all;color:#c3d2d7}</style></head><body><main><div>REO MUSIC · Android 8+</div><h1>REO Pa600QT Remote v4</h1><p>تطبيق أصلي مُوسّع لـ Pa600QT عبر USB-C OTG على Galaxy A52s. يتضمن لوحة أفقية تشبه ترتيب Pa600QT، أزرار Intro وVariation وEnding وFills وSTS، PAD التجريبي، إدارة Styles وSongBook والأصوات ومراقبة MIDI.</p><a href="./REO-Pa600QT-v4-A52s.apk" download>تحميل تطبيق Android APK</a><p class="small">هذه نسخة تجريبية موقعة بتوقيع Android Debug. نجاح البناء لا يثبت التشغيل على هاتف حقيقي. PAD 1–4 تجارب لمسارات MIDI ولا تمثل الضغط المضمون على PAD الداخلي. قد يعرض أندرويد تحذيراً عند تثبيت ملف APK من خارج المتجر.</p><p class="small">للتحقق من سلامة الملف، راجع بصمة SHA-256:</p><code id="sha"></code><p class="small"><a style="background:none;color:#aedacc;text-align:right;padding:0;margin:0;font-size:13px" href="./SHA256.txt">ملف بصمة SHA-256</a></p></main><script>fetch("./SHA256.txt").then(x=>x.text()).then(x=>{document.getElementById("sha").textContent=x.split(/\s+/)[0]}).catch(()=>{});</script></body></html>
HTML
cat > dist/vercel.json <<'JSON'
{"headers":[{"source":"/(.*)\\.apk","headers":[{"key":"Content-Type","value":"application/vnd.android.package-archive"},{"key":"Content-Disposition","value":"attachment; filename=REO-Pa600QT-v4-A52s.apk"}]}]}
JSON

echo "APK build succeeded"
