# 相机预览与目标检测性能优化说明

> **当前工程状态**：已回退为 **`DnnDetector::detectAndDraw`** 在 RGBA 上绘制框与文字，再经 `updateCameraFrame` 上传纹理；下文中的 `detectOnly`、OpenGL 线框叠加、`DetectionTypes.h` 等路径**已不再存在于代码中**，仅作历史记录。

本文记录针对「手持移动时画面刷新卡顿」所做的优化思路与实现要点。

## 现象

识别与画框功能可用后，晃动或移动手机时，视频流观感明显掉帧、刷新滞后。

## 根因分析

1. **分析线程负载过重**  
   每帧在 `ImageAnalysis` 回调里执行：整图 `RGBA→BGR`、YOLO 推理、NMS、再在整幅 RGBA 上做两次 `cvtColor`（BGRA 中转）+ `rectangle`/`putText`。单帧耗时长会导致 `STRATEGY_KEEP_ONLY_LATEST` 大量丢帧，预览与检测都难跟上。

2. **GPU 纹理路径偏重**  
   每帧使用 `glTexImage2D` 上传整幅纹理，会触发驱动侧更重的分配/同步；尺寸不变时更适合 `glTexSubImage2D`。

3. **Java 侧拷贝**  
   在 `rowStride == width * 4` 时仍按行 `get/put`，增加不必要循环与 JNI 交互。

4. **分辨率过高**  
   分析用全分辨率会线性放大内存带宽与 DNN 前处理成本。

## 优化措施

### 1. 推理与「画框」解耦（核心）

- **之前**：`detectAndDraw` 在相机图像上原地绘制框和文字，再交给 GL 显示。  
- **之后**：  
  - `DnnDetector::detectOnly()` **只读**图像，完成推理后写入归一化框 `NormDetection`（相对图像宽高的 0–1 坐标）。  
  - **非推理帧**直接返回，不做颜色转换、不跑网络。  
  - `Renderer::setDetectionOverlay()` 接收框列表，在 **OpenGL** 中用 `GL_LINE_LOOP` + 独立 line shader 绘制，变换与全屏相机纹理一致（同一套 `sx/sy` 与旋转角）。  

效果：相机线程大部分帧只做拷贝与上传；重活集中在间隔帧，预览流畅度明显提升。  

代价：当前实现 **不再在 CPU 上绘制英文标签**（原路径两次全图 `cvtColor` 成本过高）。若需文字，可后续考虑：低频小区域纹理、或简化 GL 字模。

### 2. 降低推理调用频率

- 成员 `inferEveryN_`（如设为 **3**）：每 N 帧才跑一次 DNN，中间帧复用上一帧框。  
- 可调：更小 → 框更跟手但更费 CPU；更大 → 更省电、更流畅，框略滞后。

### 3. 纹理上传

- 宽高与上一帧一致且纹理已创建时，使用 **`glTexSubImage2D`**；尺寸变化时仍用 **`glTexImage2D`** 并更新记录的宽高。

### 4. CameraX 分析分辨率

- `ImageAnalysis.Builder.setTargetResolution(new Size(960, 540))`（示例），由设备选取最接近的可用分辨率。  
- 可按画质/性能再改为 `1280×720` 或更低。

### 5. Java 快速拷贝路径

- 当 `plane.getRowStride() == width * 4` 时，对 `ByteBuffer` **单次 `put` 整块**，否则保持按行拷贝以处理 stride 大于行宽的情况。

### 6. JNI 调用顺序

- 先 `detectOnly`（只读），再 `updateCameraFrame`（拷贝纯画面到渲染缓冲），最后 `setDetectionOverlay`，保证上传纹理为未污染的相机内容，框仅由 GL 叠加。

## 涉及文件（便于维护）

| 区域 | 文件 |
|------|------|
| 归一化检测框结构 | `app/src/main/cpp/DetectionTypes.h` |
| DNN 仅推理 | `app/src/main/cpp/DnnDetector.{h,cpp}` |
| GL 线框 + 纹理子图上传 | `app/src/main/cpp/Renderer.{h,cpp}` |
| JNI 顺序与 overlay | `app/src/main/cpp/main.cpp` |
| 分辨率与 Buffer 拷贝 | `app/src/main/java/.../MainActivity.java` |

---

## 附录：关键实现代码（与当前工程一致）

以下摘录便于对照；若你后续改过行号，以仓库源文件为准。

### `DetectionTypes.h` — 归一化框

```cpp
// 归一化检测框：相对相机图像左上角，宽高归一化到 [0,1]
struct NormDetection {
    float nx = 0.f;
    float ny = 0.f;
    float nw = 0.f;
    float nh = 0.f;
};
```

### `DnnDetector.h` — 仅推理接口与降频参数

```cpp
// YOLOv5 ONNX：仅推理，不修改图像；框由 Renderer 用 GL 绘制
void detectOnly(const uint8_t* rgba, int width, int height);
const std::vector<NormDetection>& getLastDetections() const { return lastNorm_; }
// ...
int inferEveryN_ = 3;
std::vector<NormDetection> lastNorm_;
```

### `DnnDetector.cpp` — 非推理帧早退 + NMS 后写入 `lastNorm_`

```cpp
void DnnDetector::detectOnly(const uint8_t* rgba, int width, int height) {
    if (!loaded_ || width <= 0 || height <= 0) return;

    if ((frameCounter_++ % inferEveryN_) != 0) {
        return;
    }

    cv::Mat rgbaMat(height, width, CV_8UC4, const_cast<uint8_t*>(rgba));
    cv::Mat bgr;
    cv::cvtColor(rgbaMat, bgr, cv::COLOR_RGBA2BGR);
    // ... letterbox、blobFromImage、forward、解析行、NMS ...
    lastNorm_.clear();
    const float invW = 1.f / static_cast<float>(width);
    const float invH = 1.f / static_cast<float>(height);
    for (int idx : indices) {
        const auto& b = boxes[static_cast<size_t>(idx)];
        lastNorm_.push_back(NormDetection{b.x * invW, b.y * invH,
                                         static_cast<float>(b.width) * invW,
                                         static_cast<float>(b.height) * invH});
    }
}
```

### `main.cpp` — JNI：先推理（只读），再上传纹理，再下发 overlay

```cpp
if (g_dnn.isLoaded()) {
    g_dnn.detectOnly(reinterpret_cast<const uint8_t*>(addr), width, height);
}

if (g_renderer) {
    g_renderer->updateCameraFrame(reinterpret_cast<const uint8_t*>(addr), width, height, rotationDegrees);
    if (g_dnn.isLoaded()) {
        g_renderer->setDetectionOverlay(g_dnn.getLastDetections());
    } else {
        g_renderer->setDetectionOverlay(std::vector<NormDetection>{});
    }
}
```

### `MainActivity.java` — 分析分辨率 + 紧凑 stride 快速拷贝

```java
ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setTargetRotation(displayRotation)
        .setTargetResolution(new Size(960, 540))
        .build();

// ...
mDirectBuffer.clear();
if (rowStride == width * 4) {
    buffer.position(0);
    buffer.limit(height * rowStride);
    mDirectBuffer.put(buffer);
    buffer.clear();
} else {
    for (int row = 0; row < height; row++) {
        buffer.position(row * rowStride);
        buffer.get(mRowData);
        mDirectBuffer.put(mRowData);
    }
}
```

### `Renderer.h` — overlay 与纹理尺寸记录

```cpp
void setDetectionOverlay(const std::vector<NormDetection>& boxes);
// ...
std::unique_ptr<Shader> lineShader_;
int cameraTexUploadW_ = 0;
int cameraTexUploadH_ = 0;
std::vector<NormDetection> overlayBoxes_;
std::mutex overlayMutex_;
GLuint lineVbo_ = 0;
```

### `Renderer.cpp` — 像素坐标映射到与背景一致的 clip 空间

```cpp
inline void mapPixelToClip(float px, float py, float imgW, float imgH, float sx, float sy,
                           float cosA, float sinA, float* outX, float* outY) {
    const float lx = (px / imgW - 0.5f) * 2.0f * sx;
    const float ly = sy * (1.0f - 2.0f * (py / imgH));
    *outX = cosA * lx + sinA * ly;
    *outY = -sinA * lx + cosA * ly;
}
```

### `Renderer.cpp` — 同尺寸用 `glTexSubImage2D`

```cpp
if (cameraTexture_ != 0 && cameraTexUploadW_ == cameraWidth_ && cameraTexUploadH_ == cameraHeight_) {
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cameraWidth_, cameraHeight_, GL_RGBA, GL_UNSIGNED_BYTE,
                    cameraDataBuffer_.data());
} else {
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, cameraWidth_, cameraHeight_, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 cameraDataBuffer_.data());
    cameraTexUploadW_ = cameraWidth_;
    cameraTexUploadH_ = cameraHeight_;
}
```

### `Renderer.cpp` — 线框专用 shader（与相机 shader 分开展示）

```glsl
// lineVertex
#version 300 es
in vec3 inPosition;
in vec2 inUV;
uniform mat4 uProjection;
void main() {
    gl_Position = uProjection * vec4(inPosition, 1.0);
}

// lineFragment
#version 300 es
precision mediump float;
out vec4 outColor;
void main() {
    outColor = vec4(0.0, 0.92, 0.28, 1.0);
}
```

初始化与 VBO：

```cpp
lineShader_ = std::unique_ptr<Shader>(Shader::loadShader(lineVertex, lineFragment, "inPosition", "inUV", "uProjection"));
glGenBuffers(1, &lineVbo_);
```

### `Renderer.cpp` — 绘制 overlay（锁内拷贝框列表，逐框 `GL_LINE_LOOP`）

逻辑要点（完整循环见源文件）：在画完背景纹理后，`lineShader_` + 单位矩阵；对每个 `NormDetection` 将四角像素经 `mapPixelToClip` 写入 VBO，`glDrawArrays(GL_LINE_LOOP, 0, 4)`。

```cpp
void Renderer::setDetectionOverlay(const std::vector<NormDetection>& boxes) {
    std::lock_guard<std::mutex> lk(overlayMutex_);
    overlayBoxes_ = boxes;
}

// render() 内，背景 drawModel 之后：
if (lineShader_ && lineVbo_ != 0) {
    std::vector<NormDetection> boxesCopy;
    {
        std::lock_guard<std::mutex> lk(overlayMutex_);
        boxesCopy = overlayBoxes_;
    }
    if (!boxesCopy.empty()) {
        // ... lineShader_->activate()，identity uProjection，glBindBuffer(lineVbo_)
        // for (d : boxesCopy) 填四角顶点、glBufferData(GL_STREAM_DRAW)、glDrawArrays(GL_LINE_LOOP, 0, 4)
    }
}
```

---

## 后续可扩展方向

- **NNAPI / GPU** 后端：在 `DnnDetector::loadModel` 中尝试 OpenCV DNN 的 `DNN_TARGET_OPENCL` / 厂商 NPU（需设备与库支持）。  
- **双分辨率**：低分辨率专供分析，显示仍用较高分辨率（需绑定多路用例或缩放策略）。  
- **PBO 异步上传**：进一步降低 CPU→GPU 拷贝与绘制线程的耦合（改动较大）。  

---

*文档与工程内优化实现同步记录，修改参数时建议同步更新本节「可调」说明。*
