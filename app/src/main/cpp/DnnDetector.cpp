#include "DnnDetector.h"
#include "AndroidOut.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>

namespace {
static constexpr int kNumClasses = 80;
static const char* kCocoNames[kNumClasses] = {
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
};

inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}
} // namespace

DnnDetector::DnnDetector() {
    running_ = true;
    workerThread_ = std::thread(&DnnDetector::inferenceLoop, this);
}

DnnDetector::~DnnDetector() {
    running_ = false;
    cv_.notify_all();
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
}

bool DnnDetector::loadModel(const std::string& onnxPath) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnxPath);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        // 移动端 GPU：需 OpenCV Android SDK 带 OpenCL；不支持时可改回 DNN_TARGET_CPU
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL_FP16);
        loaded_ = !net_.empty();
        if (loaded_) {
            aout << "DnnDetector: loaded " << onnxPath << std::endl;
        }
        return loaded_;
    } catch (const cv::Exception& e) {
        aout << "DnnDetector load exception: " << e.what() << std::endl;
        loaded_ = false;
    }
    return false;
}

cv::Mat DnnDetector::letterbox(const cv::Mat& bgr, int targetSize, Letterbox& lb) {
    int w = bgr.cols;
    int h = bgr.rows;
    // 等比例缩放
    lb.scale = std::min(static_cast<float>(targetSize) / static_cast<float>(w),
                        static_cast<float>(targetSize) / static_cast<float>(h));
    int nw = cvRound(static_cast<float>(w) * lb.scale);
    int nh = cvRound(static_cast<float>(h) * lb.scale);
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(nw, nh));
    int dw = targetSize - nw;
    int dh = targetSize - nh;
    lb.padLeft = dw / 2;
    lb.padTop = dh / 2;
    // 填充灰色，目标尺寸640 * 640
    cv::Mat out(targetSize, targetSize, bgr.type(), cv::Scalar(114, 114, 114));
    // 将缩放后的图像复制到目标图像的相应位置
    // cv::Mat 就提供了多种 operator() 重载，其中一种形式是：
    // Mat operator()( const Rect& roi ) const; 用于获取子图像
    cv::Mat roi = out(cv::Rect(lb.padLeft, lb.padTop, nw, nh));
    resized.copyTo(roi);
    return out; // 返回目标图像
}

void DnnDetector::inferenceLoop() {
    while (running_) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return !running_ || hasNewFrame_; });
            if (!running_) break;
            frame = pendingFrame_.clone();
            hasNewFrame_ = false;
        }

        if (frame.empty() || !loaded_) continue;

        // 推理开始
        cv::Mat bgr;
        cv::cvtColor(frame, bgr, cv::COLOR_RGBA2BGR);

        Letterbox lb;
        cv::Mat padded = letterbox(bgr, inputSize_, lb);
        cv::Mat blob = cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(inputSize_, inputSize_),
                                               cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat out;
        try {
            out = net_.forward();
        } catch (const cv::Exception& e) {
            aout << "DnnDetector forward exception: " << e.what() << std::endl;
            continue;
        }

        cv::Mat out2d;
        if (out.dims == 3 && out.size[0] == 1) {
            //所以条件 out.dims == 3 && out.size[0] == 1 的意思是：
// 输出是 3 维张量，且 batch = 1，可以安全地把它看成「去掉 batch 维之后」的 N × C 二维表
            out2d = cv::Mat(out.size[1], out.size[2], CV_32F, out.ptr<float>());
        } else if (out.dims == 2) {
            out2d = out;
        } else continue;

        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> classIds;

        for (int i = 0; i < out2d.rows; ++i) {
            const float* row = out2d.ptr<float>(i);
            float obj = row[4];
            if (obj > 1.f || obj < 0.f) obj = sigmoid(obj);
            if (obj < confThresh_) continue;

            float bestProb = row[5];
            int bestCls = 0;
            for (int c = 1; c < kNumClasses; ++c) {
                if (row[5 + c] > bestProb) {
                    bestProb = row[5 + c];
                    bestCls = c;
                }
            }
            if (bestProb > 1.f || bestProb < 0.f) bestProb = sigmoid(bestProb);
            float conf = obj * bestProb;
            if (conf < confThresh_) continue;

            float cx = row[0], cy = row[1], bw = row[2], bh = row[3];
            float x1 = (cx - bw * 0.5f - lb.padLeft) / lb.scale;
            float y1 = (cy - bh * 0.5f - lb.padTop) / lb.scale;
            boxes.emplace_back(x1, y1, bw / lb.scale, bh / lb.scale);
            scores.push_back(conf);
            classIds.push_back(bestCls);
        }

        std::vector<int> indices;
        std::vector<cv::Rect> boxesI;
        for (const auto& b : boxes) {
            boxesI.emplace_back(cvRound(b.x), cvRound(b.y), cvRound(b.width), cvRound(b.height));
        }
        /*
        大致流程是：

按 scores（置信度）从高到低排序。
先保留分最高的框，再看后面的框和它 IoU（交并比） 有多大。
若与已保留框的 IoU 超过 nmsThresh_（你头文件里一般是 0.45 这类值），就认为是在框同一个目标，删掉这个低分框。
重复直到处理完所有框。
输出 indices：保留下来的框在 boxesI / scores / classIds 里的下标，后面你用这些下标去填 lastBoxes_ 等。
        
        */
        cv::dnn::NMSBoxes(boxesI, scores, 0.01f, nmsThresh_, indices);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            lastBoxes_.clear();
            lastClassIds_.clear();
            lastConfs_.clear();
            for (int idx : indices) {
                lastBoxes_.push_back(boxes[idx]);
                lastClassIds_.push_back(classIds[idx]);
                lastConfs_.push_back(scores[idx]);
            }
        }
    }
}

void DnnDetector::detectAndDraw(uint8_t* rgba, int width, int height) {
    if (!loaded_ || width <= 0 || height <= 0) return;

    cv::Mat rgbaMat(height, width, CV_8UC4, rgba);

    // 尝试推送新帧
    if (mtx_.try_lock()) {
        if (!hasNewFrame_) {
            rgbaMat.copyTo(pendingFrame_);
            hasNewFrame_ = true;
            cv_.notify_one();
        }
        mtx_.unlock();
    }

    // 绘制旧结果
    std::vector<cv::Rect2d> boxes;
    std::vector<int> cids;
    std::vector<float> confs;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        boxes = lastBoxes_;
        cids = lastClassIds_;
        confs = lastConfs_;
    }

    if (!boxes.empty()) {
        drawLabelsRgba(rgbaMat, boxes, cids, confs);
    }
}

void DnnDetector::drawLabelsRgba(cv::Mat& rgba, const std::vector<cv::Rect2d>& boxes,
                                 const std::vector<int>& classIds, const std::vector<float>& confs) {
    const int thickness = 2;
    const double fontScale = 0.5;
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& r = boxes[i];
        cv::Point p1(cvRound(r.x), cvRound(r.y));
        cv::Point p2(cvRound(r.x + r.width), cvRound(r.y + r.height));
        cv::rectangle(rgba, p1, p2, cv::Scalar(0, 255, 0, 255), thickness);

        int cid = classIds[i];
        const char* name = (cid >= 0 && cid < kNumClasses) ? kCocoNames[cid] : "?";
        std::string text = std::string(name) + " " + std::to_string(static_cast<int>(confs[i] * 100)) + "%";

        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline);
        int ty = std::max(p1.y - 4, ts.height + 2);
        cv::rectangle(rgba, cv::Point(p1.x, ty - ts.height - 4), cv::Point(p1.x + ts.width + 4, ty + 2),
                      cv::Scalar(0, 0, 0, 150), cv::FILLED);
        cv::putText(rgba, text, cv::Point(p1.x + 2, ty), cv::FONT_HERSHEY_SIMPLEX,
                    fontScale, cv::Scalar(0, 255, 0, 255), 1, cv::LINE_AA);
    }
}
