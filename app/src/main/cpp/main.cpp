#include <jni.h>

#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <game-activity/GameActivity.h>

#include "AndroidOut.h"
#include "DnnDetector.h"
#include "Renderer.h"

#include <string>

static Renderer* g_renderer = nullptr;
static DnnDetector g_dnn;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_example_game_1test_MainActivity_nativeInitDnn(JNIEnv* env, jclass /*clazz*/, jstring jpath) {
    if (!jpath) return JNI_FALSE;
    const char* utf = env->GetStringUTFChars(jpath, nullptr);
    if (!utf) return JNI_FALSE;
    const bool ok = g_dnn.loadModel(std::string(utf));
    env->ReleaseStringUTFChars(jpath, utf);
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_example_game_1test_MainActivity_processCameraFrameDirect(JNIEnv *env, jobject thiz, jobject buffer,
                                                                jint width, jint height, jint rotationDegrees) {
    void* addr = env->GetDirectBufferAddress(buffer);
    if (!addr) return;

    if (g_dnn.isLoaded()) {
        g_dnn.detectAndDraw(reinterpret_cast<uint8_t*>(addr), width, height);
    }

    if (g_renderer) {
        g_renderer->updateCameraFrame(reinterpret_cast<const uint8_t*>(addr), width, height, rotationDegrees);
    }
}

void handle_cmd(android_app *pApp, int32_t cmd) {
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            if (!pApp->userData) {
                g_renderer = new Renderer(pApp);
                pApp->userData = g_renderer;
            }
            break;
        case APP_CMD_TERM_WINDOW:
            if (pApp->userData) {
                auto *pRenderer = reinterpret_cast<Renderer *>(pApp->userData);
                delete pRenderer;
                pApp->userData = nullptr;
                g_renderer = nullptr;
            }
            break;
        default:
            break;
    }
}

bool motion_event_filter_func(const GameActivityMotionEvent *motionEvent) {
    auto sourceClass = motionEvent->source & AINPUT_SOURCE_CLASS_MASK;
    return (sourceClass == AINPUT_SOURCE_CLASS_POINTER ||
            sourceClass == AINPUT_SOURCE_CLASS_JOYSTICK);
}

void android_main(struct android_app *pApp) {
    pApp->onAppCmd = handle_cmd;
    android_app_set_motion_event_filter(pApp, motion_event_filter_func);

    while (!pApp->destroyRequested) {
        int events;
        android_poll_source *pSource;
        while (ALooper_pollOnce(0, nullptr, &events, (void**)&pSource) >= 0) {
            if (pSource) pSource->process(pApp, pSource);
        }

        if (pApp->userData) {
            auto *pRenderer = reinterpret_cast<Renderer *>(pApp->userData);
            pRenderer->handleInput();
            pRenderer->render();
        }
    }
}
}
