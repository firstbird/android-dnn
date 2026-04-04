#ifndef GAME_TEST_DNN_DETECTOR_H
#define GAME_TEST_DNN_DETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

// YOLOv5 ONNX：推理后在 RGBA 帧上绘制框与 COCO 英文标签（与 GL 纹理一致，避免线框错位）
class DnnDetector {
public:
    bool loadModel(const std::string& onnxPath);
    bool isLoaded() const { return loaded_; }

    void detectAndDraw(uint8_t* rgba, int width, int height);

private:
    struct Letterbox {
        float scale = 1.f;
        int padLeft = 0;
        int padTop = 0;
    };

    static cv::Mat letterbox(const cv::Mat& bgr, int targetSize, Letterbox& lb);
    void drawLabelsRgba(cv::Mat& rgba, const std::vector<cv::Rect2d>& boxes,
                        const std::vector<int>& classIds, const std::vector<float>& confs);

    cv::dnn::Net net_;
    bool loaded_ = false;
    int inputSize_ = 640;
    float confThresh_ = 0.35f;
    float nmsThresh_ = 0.45f;
    int frameCounter_ = 0;
    int inferEveryN_ = 2;
    std::vector<cv::Rect2d> lastBoxes_;
    std::vector<int> lastClassIds_;
    std::vector<float> lastConfs_;
    cv::Mat bgraScratch_;
};

#endif
