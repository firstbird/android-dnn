import java.io.File
import java.net.URI
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
}

val localProperties = Properties().apply {
    rootProject.file("local.properties").takeIf { it.exists() }?.reader()?.use { load(it) }
}
val opencvSdkProperty = localProperties.getProperty("opencv.dir")?.trim()
val opencvRoot = rootProject.file(opencvSdkProperty ?: "opencv/OpenCV-android-sdk")

// 与官方发布包一致：https://github.com/opencv/opencv/releases
val openCvReleaseVersion = "4.10.0"
val openCvAndroidSdkZipUrl =
    "https://github.com/opencv/opencv/releases/download/$openCvReleaseVersion/opencv-$openCvReleaseVersion-android-sdk.zip"
val openCvConfigMarker = File(opencvRoot, "sdk/native/jni/OpenCVConfig.cmake")

tasks.register("ensureOpenCvAndroidSdk") {
    description = "若缺少 OpenCV Android SDK，则下载并解压到工程 opencv/ 目录"
    group = "build setup"
    onlyIf { !openCvConfigMarker.exists() }
    doLast {
        if (opencvSdkProperty != null) {
            throw GradleException(
                "local.properties 中 opencv.dir 已设为「$opencvSdkProperty」，但该路径下没有 sdk/native/jni/OpenCVConfig.cmake。\n" +
                    "请改为正确的 OpenCV-android-sdk 根目录，或删除 opencv.dir 以使用默认路径并自动下载。"
            )
        }
        if (gradle.startParameter.isOffline) {
            throw GradleException(
                "未找到 OpenCV SDK 且当前为离线模式。请联网构建一次，或手动解压官方 opencv-*-android-sdk.zip 到：\n" +
                    "${opencvRoot.absolutePath}"
            )
        }
        val parentDir = opencvRoot.parentFile ?: error("无效的 opencv 路径")
        parentDir.mkdirs()
        val zipFile = rootProject.layout.buildDirectory
            .file("downloads/opencv-$openCvReleaseVersion-android-sdk.zip")
            .get()
            .asFile
        zipFile.parentFile?.mkdirs()
        logger.lifecycle("正在下载 OpenCV Android SDK $openCvReleaseVersion（体积较大，请稍候）…")
        URI.create(openCvAndroidSdkZipUrl).toURL().openStream().use { input ->
            zipFile.outputStream().use { output -> input.copyTo(output) }
        }
        copy {
            from(rootProject.zipTree(zipFile))
            into(parentDir)
        }
        if (!openCvConfigMarker.exists()) {
            throw GradleException(
                "解压后仍未找到 OpenCVConfig.cmake，期望路径：\n${openCvConfigMarker.absolutePath}"
            )
        }
        logger.lifecycle("OpenCV Android SDK 已就绪：${opencvRoot.absolutePath}")
    }
}

android {
    namespace = "com.example.game_test"
    compileSdk {
        version = release(36) {
            minorApiLevel = 1
        }
    }

    defaultConfig {
        applicationId = "com.example.game_test"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DOPENCV_ANDROID_SDK=${opencvRoot.absolutePath}"
                )
                cppFlags += "-std=c++17"
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    sourceSets {
        getByName("main") {
            val opencvJniLibs = File(opencvRoot, "sdk/native/libs")
            if (opencvJniLibs.isDirectory) {
                jniLibs.srcDir(opencvJniLibs)
            }
        }
    }
    buildFeatures {
        prefab = true
        viewBinding = true
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

afterEvaluate {
    tasks.named("preBuild").configure { dependsOn("ensureOpenCvAndroidSdk") }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.games.activity)
    implementation(libs.camera.core)
    implementation(libs.camera.camera2)
    implementation(libs.camera.lifecycle)
    implementation(libs.camera.view)
    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}