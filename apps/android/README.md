# Jebrex AI — Android Native Client (`apps/android`)

- **Build Status on Current Machine (`2026-11-24`):** `BLOCKED` (APK not compiled locally because `gradle` and Android SDK Build-Tools are not installed and `C:` has only `6.95 GiB` free).
- **Verified Components:** Kotlin source (`MainActivity.kt`, `JebrexApiClient.kt`), `AndroidManifest.xml` (`supportsRtl="true"`, `usesCleartextTraffic="false"`, `allowBackup="false"`), and Gradle build files (`settings.gradle.kts`, `app/build.gradle.kts`).
- **Independence Guarantee:** Connects strictly to the user's local/LAN Jebrex AI REST backend (`/api/v1/auth/guest`, `/api/v1/sessions`, `/api/v1/chat`) with `use_fallback_assistant=false` and `enable_web_search=false`, displaying `response_source` transparently.

## Building on an Android Studio / Gradle Workstation
```bash
cd apps/android
gradle assembleDebug
```
