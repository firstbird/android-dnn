#ifndef ANDROIDGLINVESTIGATIONS_RENDERER_H
#define ANDROIDGLINVESTIGATIONS_RENDERER_H

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <memory>
#include <mutex>
#include <vector>

#include "Model.h"
#include "Shader.h"

struct android_app;

class Renderer {
public:
    inline Renderer(android_app *pApp) :
            app_(pApp),
            display_(EGL_NO_DISPLAY),
            surface_(EGL_NO_SURFACE),
            context_(EGL_NO_CONTEXT),
            width_(0),
            height_(0),
            shaderNeedsNewProjectionMatrix_(true),
            cameraTexture_(0),
            cameraWidth_(0),
            cameraHeight_(0),
            cameraRotation_(0),
            cameraDataUpdated_(false),
            personDetected_(false),
            personX_(0.0f), personY_(0.0f), personW_(0.0f), personH_(0.0f) {
        initRenderer();
    }

    virtual ~Renderer();

    void handleInput();
    void render();

    // 更新相机帧数据（含 OpenCV 在 RGBA 上绘制的检测框与标签）
    void updateCameraFrame(const uint8_t* data, int width, int height, int rotationDegrees);

    // 更新识别到的人物位置（坐标系：0.0-1.0）
    void updatePersonLocation(float x, float y, float w, float h) {
        personDetected_ = true;
        personX_ = x;
        personY_ = y;
        personW_ = w;
        personH_ = h;
    }

private:
    void initRenderer();
    void updateRenderArea();
    void createModels();

    android_app *app_;
    EGLDisplay display_;
    EGLSurface surface_;
    EGLContext context_;
    EGLint width_;
    EGLint height_;

    bool shaderNeedsNewProjectionMatrix_;

    std::unique_ptr<Shader> shader_;
    std::vector<Model> models_;

    // 相机渲染相关
    GLuint cameraTexture_;
    int cameraWidth_, cameraHeight_, cameraRotation_;
    std::vector<uint8_t> cameraDataBuffer_;
    bool cameraDataUpdated_;
    std::mutex cameraMutex_;

    // 人物追踪相关
    bool personDetected_;
    float personX_, personY_, personW_, personH_;
};

#endif //ANDROIDGLINVESTIGATIONS_RENDERER_H
