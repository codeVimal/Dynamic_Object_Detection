#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

// Motion detection parameters
const int minContourArea = 500;

void fillInterior(cv::Mat& frame, const std::vector<std::vector<cv::Point>>& contours)
{
    for (const auto& contour : contours)
    {
        if (cv::contourArea(contour) > minContourArea)
        {
            // Iterate through the contour points and set pixels as dynamic
            for (const auto& point : contour)
            {
                frame.at<cv::Vec3b>(point) = cv::Vec3b(0, 255, 0);  // Set pixel as dynamic (green)
            }
        }
    }
}
int main()
{
    // Video file path
    std::string videoFilePath = "E:/HPA_Problem_Statement/Problem_Statement_1/Hackathon.mp4";

    // Open the video file
    cv::VideoCapture cap(videoFilePath);

    // Check if video file is opened successfully
    if (!cap.isOpened())
    {
        std::cout << "Error opening video file." << std::endl;
        return -1;
    }

    cv::Mat frame, prevFrame, diff;

    // Creating a named window for motion detection
    cv::namedWindow("Motion Detection", cv::WINDOW_NORMAL);
    cv::resizeWindow("Motion Detection", 600, 600);

    // Read the first frame to initialize prevFrame
    if (!cap.read(prevFrame))
    {
        std::cout << "Error reading the first frame." << std::endl;
        return -1;
    }

    // Convert the first frame to grayscale
    cv::cvtColor(prevFrame, prevFrame, cv::COLOR_BGR2GRAY);

    while (cap.read(frame))
    {
        cv::Mat currentFrame;
        cv::cvtColor(frame, currentFrame, cv::COLOR_BGR2GRAY);

        // Compute absolute difference between current and previous frame
        cv::absdiff(prevFrame, currentFrame, diff);

        // Threshold the difference image
        cv::threshold(diff, diff, 30, 255, cv::THRESH_BINARY);

        // Find contours of moving objects
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(diff, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // Draw contours around moving objects
        cv::drawContours(frame, contours, -1, cv::Scalar(0, 255, 0), 2);

        // Fill the interiors of moving objects
        fillInterior(frame, contours);

        // Show the frame with motion detection
        cv::imshow("Motion Detection", frame);

        // Update prevFrame for the next iteration
        prevFrame = currentFrame.clone();

        // Break the loop if the user presses the Esc key
        if (cv::waitKey(100) == 27)
            break;
    }

    // Release video capture object
    cap.release();

    return 0;
}
