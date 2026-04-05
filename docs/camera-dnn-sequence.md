# 相机帧、渲染与 DnnDetector 异步推理时序

描述 **四条业务线程**（外加 JNI / 共享对象）如何协作：`Java UI`、`Camera 分析`、`android_main` 渲染循环、`inferenceLoop` 推理线程，以及 `DnnDetector`、`Renderer`、`OpenCV DNN Net`。

```mermaid
sequenceDiagram
    participant UI as Java UI 主线程
    participant CAM as Camera 分析线程
    participant GL as Native 渲染线程<br/>(android_main)
    participant R as Renderer
    participant D as DnnDetector
    participant W as 推理 Worker 线程
    participant N as OpenCV DNN Net

    Note over UI,W: 进程启动：System.loadLibrary；静态 g_dnn 构造时启动 W（早于或独立于 Activity）
    UI->>UI: onCreate → copyModelAndInitDnn
    UI->>D: nativeInitDnn(path)（JNI，常在 UI 线程执行）
    Note right of D: loadModel() 读 ONNX，配置 Net

    GL->>GL: GameActivity 胶水库进入 android_main
    GL->>GL: pApp->onAppCmd = handle_cmd，进入 poll 循环
    GL->>R: APP_CMD_INIT_WINDOW → new Renderer
    Note over R: g_renderer 指向该实例，供 JNI 与 render 共用

    Note over UI,N: 运行阶段：线程①②③④并行；下列三个 rect 表示三条「各自循环」，与 UI 偶发回调并存

    rect rgb(255, 248, 220)
        Note right of UI: 线程①：UI 主线程（本图仅标初始化/绑定）
        Note over UI: 权限、bindToLifecycle、Camera 绑定等（非每帧）
    end

    rect rgb(240, 240, 240)
        Note right of CAM: 线程②：ImageAnalysis 每帧 → JNI processCameraFrameDirect
        CAM->>D: detectAndDraw(rgba)
        Note over D: 1. mtx_.try_lock()
        alt 成功且 !hasNewFrame_
            D->>D: rgbaMat.copyTo(pendingFrame_)
            D->>W: cv_.notify_one()
        else 失败或已有待处理帧
            Note over D: 不覆盖 pending / 或不推送，避免阻塞 CAM
        end
        D->>D: 加锁拷贝 lastBoxes_ 等
        D->>D: drawLabelsRgba（上一批检测结果画回 RGBA）
        CAM->>R: updateCameraFrame（同一次 JNI，紧随 detectAndDraw）
        R->>R: cameraMutex_ 下 memcpy → cameraDataBuffer_
    end

    rect rgb(220, 235, 255)
        Note right of GL: 线程③：android_main 内 while (!destroyRequested)
        loop 每轮
            GL->>GL: ALooper_pollOnce（处理窗口/输入等）
            GL->>R: render()
            R->>R: 若 cameraDataUpdated_：glTexImage2D 上传纹理
            R->>R: Shader 画全屏四边形 + eglSwapBuffers
        end
    end

    rect rgb(230, 255, 230)
        Note right of W: 线程④：DnnDetector::inferenceLoop
        W->>W: cv_.wait（直至 hasNewFrame_ 或退出）
        W->>W: clone(pendingFrame_)，hasNewFrame_ = false
        W->>W: cvtColor RGBA→BGR
        W->>W: letterbox / blobFromImage
        W->>N: net_.setInput + forward()（最耗时）
        N-->>W: 检测张量
        W->>W: 解析框 + NMSBoxes
        W->>W: 加锁写 lastBoxes_ / lastClassIds_ / lastConfs_
        Note over W: 回到 wait；与 CAM、GL 无显式 join
    end
```

## 四条线程对照

| 编号 | 线程 | 典型代码位置 | 主要职责 |
|------|------|----------------|----------|
| ① | **Java UI 主线程** | `MainActivity` | 生命周期、权限、`ProcessCameraProvider.bindToLifecycle` 等；**不**跑 `ImageAnalysis` 回调 |
| ② | **Camera 分析线程** | `cameraExecutor` + `processCameraFrameDirect` | 每帧 JNI：`detectAndDraw`、`updateCameraFrame` |
| ③ | **Native 渲染线程** | `android_main`（`main.cpp`） | `ALooper_pollOnce`、`Renderer::render`、EGL 交换缓冲 |
| ④ | **推理 Worker** | `DnnDetector` 内 `std::thread` | `inferenceLoop`：`forward`、NMS、更新 `last_*` |

## 其它说明

- **DnnDetector / Renderer** 不是线程，是跨线程访问的 **C++ 对象**；同步靠 `mutex` / `condition_variable`（以及渲染侧 `cameraMutex_`）。
- **线程②** 上 **`drawLabelsRgba`** 使用的是 **已写入 `last_*` 的上一轮（或更早）结果**，与当前推送给 Worker 的帧 **不同步**，属异步折中。
- **`try_lock` 失败** 不等于「推理一定在跑」，仅表示 **当时未拿到 `mtx_`**；成功且 `hasNewFrame_ == true` 时本帧也 **不会** 再次 `copyTo(pendingFrame_)`。
- 另有多条 **系统线程**（Choreographer、Binder、GPU 驱动等），本图只覆盖与应用逻辑直接相关的四条。
