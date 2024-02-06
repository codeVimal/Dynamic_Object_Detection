#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("E:/HPA_Problem_Statement/Problem_Statement_1/Hackathon.mp4");
    cv::Ptr<cv::BackgroundSubtractorMOG2> object_detector = cv::createBackgroundSubtractorMOG2(100, 25, false);

    while (true) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret) break;

        // Object Detection
        cv::Mat mask;
        object_detector->apply(frame, mask);
        cv::threshold(mask, mask, 254, 255, cv::THRESH_BINARY);

        // Find contours and draw rectangles around moving objects
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area > 80) {
                cv::Rect bounding_box = cv::boundingRect(cnt);
                cv::rectangle(frame, bounding_box, cv::Scalar(255, 0, 0), 3);
            }
        }

        cv::imshow("Frame", frame);
        cv::imshow("Mask", mask);

        int key = cv::waitKey(100);
        if (key == 27) {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}