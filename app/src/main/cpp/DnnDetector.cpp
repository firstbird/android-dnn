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

bool DnnDetector::loadModel(const std::string& onnxPath) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnxPath);
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        loaded_ = !net_.empty();
        if (!loaded_) {
            aout << "DnnDetector: readNetFromONNX failed for " << onnxPath << std::endl;
        } else {
            aout << "DnnDetector: loaded " << onnxPath << std::endl;
        }
    } catch (const cv::Exception& e) {
        aout << "DnnDetector load exception: " << e.what() << std::endl;
        loaded_ = false;
    }
    return loaded_;
}

cv::Mat DnnDetector::letterbox(const cv::Mat& bgr, int targetSize, Letterbox& lb) {
    int w = bgr.cols;
    int h = bgr.rows;
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
    cv::Mat out(targetSize, targetSize, bgr.type(), cv::Scalar(114, 114, 114));
    cv::Mat roi = out(cv::Rect(lb.padLeft, lb.padTop, nw, nh));
    resized.copyTo(roi);
    return out;
}

void DnnDetector::drawLabelsRgba(cv::Mat& rgba, const std::vector<cv::Rect2d>& boxes,
                                 const std::vector<int>& classIds, const std::vector<float>& confs) {
    if (boxes.empty()) return;
    cv::cvtColor(rgba, bgraScratch_, cv::COLOR_RGBA2BGRA);
    const int thickness = 2;
    const double fontScale = 0.5;
    for (size_t i = 0; i < boxes.size(); ++i) {
        const auto& r = boxes[i];
        cv::Point p1(cvRound(r.x), cvRound(r.y));
        cv::Point p2(cvRound(r.x + r.width), cvRound(r.y + r.height));
        cv::rectangle(bgraScratch_, p1, p2, cv::Scalar(0, 255, 0, 255), thickness);
        int cid = classIds[i];
        const char* name = (cid >= 0 && cid < kNumClasses) ? kCocoNames[cid] : "?";
        std::ostringstream oss;
        oss << name << " " << static_cast<int>(confs[i] * 100.f) << "%";
        std::string text = oss.str();
        int baseline = 0;
        cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, fontScale, 1, &baseline);
        int ty = std::max(p1.y - 4, ts.height + 2);
        cv::rectangle(bgraScratch_,
                      cv::Point(p1.x, ty - ts.height - 4),
                      cv::Point(p1.x + ts.width + 4, ty + 2),
                      cv::Scalar(0, 0, 0, 200), cv::FILLED);
        cv::putText(bgraScratch_, text, cv::Point(p1.x + 2, ty), cv::FONT_HERSHEY_SIMPLEX,
                    fontScale, cv::Scalar(0, 255, 0, 255), 1, cv::LINE_AA);
    }
    cv::cvtColor(bgraScratch_, rgba, cv::COLOR_BGRA2RGBA);
}

void DnnDetector::detectAndDraw(uint8_t* rgba, int width, int height) {
    if (!loaded_ || width <= 0 || height <= 0) return;

    cv::Mat rgbaMat(height, width, CV_8UC4, rgba);
    cv::Mat bgr;
    cv::cvtColor(rgbaMat, bgr, cv::COLOR_RGBA2BGR);

    const bool runInfer = (frameCounter_++ % inferEveryN_) == 0;
    if (runInfer) {
        Letterbox lb;
        cv::Mat padded = letterbox(bgr, inputSize_, lb);
        cv::Mat blob =
                cv::dnn::blobFromImage(padded, 1.0 / 255.0, cv::Size(inputSize_, inputSize_),
                                       cv::Scalar(), true, false);
        net_.setInput(blob);
        cv::Mat out = net_.forward();

        cv::Mat out2d;
        if (out.dims == 3 && out.size[0] == 1) {
            const int N = out.size[1];
            const int C = out.size[2];
            if (!out.isContinuous()) out = out.clone();
            out2d = cv::Mat(N, C, CV_32F, out.ptr<float>());
        } else if (out.dims == 2) {
            out2d = out;
        } else {
            aout << "DnnDetector: unexpected output dims " << out.dims << std::endl;
            drawLabelsRgba(rgbaMat, lastBoxes_, lastClassIds_, lastConfs_);
            return;
        }

        const int rows = out2d.rows;
        const int cols = out2d.cols;
        if (cols < 5 + kNumClasses) {
            aout << "DnnDetector: unexpected output cols " << cols << std::endl;
            drawLabelsRgba(rgbaMat, lastBoxes_, lastClassIds_, lastConfs_);
            return;
        }

        std::vector<cv::Rect2d> boxes;
        std::vector<float> scores;
        std::vector<int> classIds;

        for (int i = 0; i < rows; ++i) {
            const float* row = out2d.ptr<float>(i);
            float cx = row[0];
            float cy = row[1];
            float bw = row[2];
            float bh = row[3];
            float obj = row[4];
            if (obj > 1.f || obj < 0.f) obj = sigmoid(obj);
            int bestCls = 0;
            float bestProb = row[5];
            for (int c = 1; c < kNumClasses; ++c) {
                float p = row[5 + c];
                if (p > bestProb) {
                    bestProb = p;
                    bestCls = c;
                }
            }
            if (bestProb > 1.f || bestProb < 0.f) bestProb = sigmoid(bestProb);
            float conf = obj * bestProb;
            if (conf < confThresh_) continue;

            float x1 = (cx - bw * 0.5f - static_cast<float>(lb.padLeft)) / lb.scale;
            float y1 = (cy - bh * 0.5f - static_cast<float>(lb.padTop)) / lb.scale;
            float x2 = (cx + bw * 0.5f - static_cast<float>(lb.padLeft)) / lb.scale;
            float y2 = (cy + bh * 0.5f - static_cast<float>(lb.padTop)) / lb.scale;
            x1 = std::max(0.f, std::min(x1, static_cast<float>(width - 1)));
            y1 = std::max(0.f, std::min(y1, static_cast<float>(height - 1)));
            x2 = std::max(0.f, std::min(x2, static_cast<float>(width - 1)));
            y2 = std::max(0.f, std::min(y2, static_cast<float>(height - 1)));
            if (x2 <= x1 || y2 <= y1) continue;

            boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores.push_back(conf);
            classIds.push_back(bestCls);
        }

        std::vector<int> indices;
        std::vector<cv::Rect> boxesI;
        boxesI.reserve(boxes.size());
        for (const auto& b : boxes) {
            boxesI.emplace_back(cvRound(b.x), cvRound(b.y), cvRound(b.width), cvRound(b.height));
        }
        cv::dnn::NMSBoxes(boxesI, scores, 0.01f, nmsThresh_, indices);

        lastBoxes_.clear();
        lastClassIds_.clear();
        lastConfs_.clear();
        for (int idx : indices) {
            lastBoxes_.push_back(boxes[static_cast<size_t>(idx)]);
            lastClassIds_.push_back(classIds[static_cast<size_t>(idx)]);
            lastConfs_.push_back(scores[static_cast<size_t>(idx)]);
        }
    }

    drawLabelsRgba(rgbaMat, lastBoxes_, lastClassIds_, lastConfs_);
}
