#include "Renderer.h"
#include <game-activity/native_app_glue/android_native_app_glue.h>
#include <GLES3/gl3.h>
#include <android/native_window.h>
#include <cmath>
#include <memory>
#include <vector>

#include "AndroidOut.h"
#include "Shader.h"
#include "Utility.h"

// 将背景设为红色进行调试，如果看到红色说明 GL 工作正常
#define DEBUG_RED 1.0f, 0.0f, 0.0f, 1

// 顶点着色器：每个顶点运行一次，决定顶点在屏幕上的位置，并把「纹理坐标」传给片元着色器
static const char *cameraVertex = R"vertex(#version 300 es
// OpenGL ES 3.0 着色语言版本（与 GLES3 上下文一致）

// 顶点属性：由 C++ 里 VBO + glVertexAttribPointer 传入
in vec3 inPosition;   // 顶点位置（本工程里是经投影后的裁剪空间附近坐标）
in vec2 inUV;         // 顶点对应的纹理坐标 (0~1)，用于在相机纹理上取样

// uniform：整张画面共用一个值的变量，由 C++ 的 glUniform* 设置
uniform mat4 uProjection; // 旋转等变换矩阵，把 inPosition 变到最终位置

// out：输出给「片元着色器」的同名 in（这里是插值后的 fragUV）
out vec2 fragUV;

void main() {
    // 把当前顶点的 UV 原样交给后面；三角形内部的像素会得到插值后的 UV
    fragUV = inUV;
    // gl_Position 必须是 vec4，固定表示该顶点在裁剪空间的位置（GPU 再自己做透视除法、视口变换）
    gl_Position = uProjection * vec4(inPosition, 1.0);
}
)vertex";

// 片元着色器：每个像素（片元）运行一次，决定最终颜色（这里 = 相机纹理取样结果）
static const char *cameraFragment = R"fragment(#version 300 es
// 浮点纹理采样的精度（mediump 在手机上够用且较快）
precision mediump float;

// 与顶点着色器里的 out vec2 fragUV 对应（在三角形内会自动插值）
in vec2 fragUV;

uniform sampler2D uTexture; // 绑定到纹理单元上的相机画面（RGBA）

// 由 C++ 传入 (0或1, 0或1)：为 1 时在对应轴做镜像，修正竖屏预览方向/后置镜像
uniform vec2 uTexFlip;

out vec4 outColor; // 输出到帧缓冲的颜色 (RGBA)

void main() {
    // mix(a, b, t) = a*(1-t)+b*t：t 为 0 保持原 UV，为 1 则变成 1-uv，即沿该轴翻转取样
    vec2 uv = vec2(
        mix(fragUV.x, 1.0 - fragUV.x, uTexFlip.x),
        mix(fragUV.y, 1.0 - fragUV.y, uTexFlip.y));
    // 用 uv 在相机纹理上取颜色，作为该像素最终颜色
    outColor = texture(uTexture, uv);
}
)fragment";

Renderer::~Renderer() {
    if (display_ != EGL_NO_DISPLAY) {
        eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
        if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
        eglTerminate(display_);
    }
}

void Renderer::updateCameraFrame(const uint8_t* data, int width, int height, int rotationDegrees) {
    std::lock_guard<std::mutex> lock(cameraMutex_);
    cameraWidth_ = width;
    cameraHeight_ = height;
    cameraRotation_ = rotationDegrees;
    size_t size = width * height * 4;
    if (cameraDataBuffer_.size() != size) cameraDataBuffer_.resize(size);
    memcpy(cameraDataBuffer_.data(), data, size);
    cameraDataUpdated_ = true;
}

void Renderer::render() {
    if (display_ == EGL_NO_DISPLAY || surface_ == EGL_NO_SURFACE) return;

    updateRenderArea();

    // 每一帧都清屏为黑色
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 1. 更新相机纹理
    if (cameraDataUpdated_) {
        std::lock_guard<std::mutex> lock(cameraMutex_);
        if (cameraTexture_ == 0) {
            glGenTextures(1, &cameraTexture_);
            glBindTexture(GL_TEXTURE_2D, cameraTexture_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        glBindTexture(GL_TEXTURE_2D, cameraTexture_);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cameraWidth_, cameraHeight_, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     cameraDataBuffer_.data());
        cameraDataUpdated_ = false;
    }

    if (shader_ && cameraTexture_ != 0) {
        shader_->activate();

        // 设置纹理单元
        GLint texLoc = glGetUniformLocation(shader_->getProgram(), "uTexture");
        glUniform1i(texLoc, 0);

        // 竖屏(90/270)：需修正 GL 行序(Y) + 后置摄像头预览左右镜像(X)。横屏(0/180)：仅旋转即可，再全局翻 Y 会上下颠倒
        const bool portraitRotation =
                cameraRotation_ == 90 || cameraRotation_ == 270;
        GLint texFlipLoc = glGetUniformLocation(shader_->getProgram(), "uTexFlip");
        if (texFlipLoc >= 0) {
            glUniform2f(texFlipLoc, portraitRotation ? 1.0f : 0.0f, portraitRotation ? 1.0f : 0.0f);
        }

        // CameraX：顺时针旋转多少度可使画面直立；顶点绕 Z 逆时针为正，取负与预览方向一致
        float angleRad = -(float)cameraRotation_ * static_cast<float>(M_PI) / 180.0f;
        // 竖屏时在设备上整体再顺时针 180° 才与肉眼一致（等价于此处再叠加 π）
        if (portraitRotation) {
            angleRad += static_cast<float>(M_PI);
        }
        float cosA = cosf(angleRad);
        float sinA = sinf(angleRad);

        float rotationMatrix[16] = {
            cosA,  sinA, 0, 0,
            -sinA, cosA, 0, 0,
            0,     0,    1, 0,
            0,     0,    0, 1
        };

        shader_->setProjectionMatrix(rotationMatrix);

        // 按「旋转后的」逻辑宽高比做 contain，避免竖屏时被拉成正方形
        float sa = width_ > 0 && height_ > 0 ? (float)width_ / (float)height_ : 1.0f;
        bool swapDims = (cameraRotation_ == 90 || cameraRotation_ == 270);
        float logicW = swapDims ? (float)cameraHeight_ : (float)cameraWidth_;
        float logicH = swapDims ? (float)cameraWidth_ : (float)cameraHeight_;
        float ia = logicH > 0 ? logicW / logicH : 1.0f;
        float sx = 1.0f;
        float sy = 1.0f;
        if (sa > ia) {
            sx = ia / sa;
        } else if (ia > 0) {
            sy = sa / ia;
        }

        std::vector<Vertex> bgVerts = {
            Vertex(Vector3{ sx,  sy, 0.0f}, Vector2{1.0f, 0.0f}),
            Vertex(Vector3{-sx,  sy, 0.0f}, Vector2{0.0f, 0.0f}),
            Vertex(Vector3{-sx, -sy, 0.0f}, Vector2{0.0f, 1.0f}),
            Vertex(Vector3{ sx, -sy, 0.0f}, Vector2{1.0f, 1.0f})
        };
        std::vector<Index> indices = {0, 1, 2, 0, 2, 3};

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, cameraTexture_);

        Model bgModel(bgVerts, indices, nullptr);
        shader_->drawModel(bgModel);

        // 绘制识别方块
        if (personDetected_) {
            float x = personX_ * 2.0f - 1.0f;
            float y = 1.0f - personY_ * 2.0f;
            float w = personW_;
            float h = personH_;
            std::vector<Vertex> boxVerts = {
                Vertex(Vector3{x+w, y+h, 0.1f}, Vector2{1,1}),
                Vertex(Vector3{x-w, y+h, 0.1f}, Vector2{0,1}),
                Vertex(Vector3{x-w, y-h, 0.1f}, Vector2{0,0}),
                Vertex(Vector3{x+w, y-h, 0.1f}, Vector2{1,0})
            };
            Model boxModel(boxVerts, indices, nullptr);
            shader_->drawModel(boxModel);
        }
    } else {
        // 如果没有纹理或 Shader，至少能看到红色背景
    }

    eglSwapBuffers(display_, surface_);
}

void Renderer::initRenderer() {
    aout << "Initializing Renderer..." << std::endl;

    EGLint attribs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_STENCIL_SIZE, 8,
        EGL_NONE
    };

    auto display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(display, nullptr, nullptr);

    EGLint numConfigs;
    eglChooseConfig(display, attribs, nullptr, 0, &numConfigs);
    std::unique_ptr<EGLConfig[]> configs(new EGLConfig[numConfigs]);
    eglChooseConfig(display, attribs, configs.get(), numConfigs, &numConfigs);

    if (numConfigs <= 0) {
        aout << "No EGL configurations found!" << std::endl;
        return;
    }

    auto config = configs[0];

    // 必须在创建 Surface 前设置窗口格式
    ANativeWindow_setBuffersGeometry(app_->window, 0, 0, WINDOW_FORMAT_RGBA_8888);

    surface_ = eglCreateWindowSurface(display, config, app_->window, nullptr);
    if (surface_ == EGL_NO_SURFACE) {
        aout << "Failed to create EGL surface, error: " << eglGetError() << std::endl;
        return;
    }

    EGLint contextAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    context_ = eglCreateContext(display, config, nullptr, contextAttribs);
    if (context_ == EGL_NO_CONTEXT) {
        aout << "Failed to create EGL context, error: " << eglGetError() << std::endl;
        return;
    }

    if (eglMakeCurrent(display, surface_, surface_, context_) == EGL_FALSE) {
        aout << "eglMakeCurrent failed" << std::endl;
        return;
    }

    display_ = display;

    // 加载 Shader 并捕获可能的错误
    shader_ = std::unique_ptr<Shader>(Shader::loadShader(cameraVertex, cameraFragment, "inPosition", "inUV", "uProjection"));
    if (!shader_) {
        aout << "Shader compilation FAILED!" << std::endl;
    } else {
        aout << "Shader compiled successfully." << std::endl;
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::updateRenderArea() {
    EGLint w, h;
    eglQuerySurface(display_, surface_, EGL_WIDTH, &w);
    eglQuerySurface(display_, surface_, EGL_HEIGHT, &h);
    if (w > 0 && h > 0 && (w != width_ || h != height_)) {
        width_ = w; height_ = h;
        glViewport(0, 0, w, h);
    }
}

void Renderer::handleInput() {
    auto *buf = android_app_swap_input_buffers(app_);
    if (buf) android_app_clear_motion_events(buf);
}

void Renderer::createModels() {}
