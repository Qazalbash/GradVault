#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

#define WHITE cv::Vec3b(255, 255, 255)  // white
#define GREEN cv::Vec3b(0, 255, 0)      // green

#define SHOW_GALLERY 0

// directions for the algorithm
constexpr int directions[16] = {
    -1, 0,   // up
    -1, -1,  // up left
    0,  -1,  // left
    1,  -1,  // down left
    1,  0,   // down
    1,  1,   // down right
    0,  1,   // right
    -1, 1    // up right
};

/**
 * @brief Radial Sweep Algorithm for boundary tracing
 *
 * @param img
 * @return cv::Mat*
 */
cv::Mat radial_sweep_algorithm(const cv::Mat& img) {
    const int height = img.rows, width = img.cols;

    int y = 0, x = 0;

    // find the first white cv::Vec3b
    for (int k = 0; k < width * height; ++k) {
        y = k / width;
        x = k % width;
        if (img.at<cv::Vec3b>(y, x) == WHITE) {
            break;
        }
    }

    // coordinates of the first white cv::Vec3b that are starting point of the boundary
    const int start_x = x, start_y = y;

    int pos = 0;  // position of the current direction in the directions vector

    std::vector<int> boundary_points = {y, x};  // boundary points

    const int n = sizeof(directions) / sizeof(int);

    do {
        // loop over all directions
        for (int i = 0; i < n; ++i) {
            // next coordinates
            const int x_ = x + directions[pos], y_ = y + directions[pos + 1];
            if (0 > x_ || x_ >= width || 0 > y_ || y_ >= height || img.at<cv::Vec3b>(y_, x_) != WHITE) {
                // if the next cv::Vec3b is not white or out of bounds go to the next direction
                pos = (pos + 2) % n;
            } else {
                // if the next cv::Vec3b is white, go to the opposite direction, add the current coordinates to the
                // boundary points, update the current coordinates to the next coordinates, and break the loop
                pos = (pos + 10) % n;
                boundary_points.push_back(y);
                boundary_points.push_back(x);
                x = x_;
                y = y_;
                break;
            }
        }
        // if the current coordinates are the same as the starting coordinates break the loop
    } while (x != start_x || y != start_y);

    cv::Mat bw_img = img.clone();

    // draw the boundary points
    for (size_t i = 0ul; i < boundary_points.size(); i += 2ul) {
        const int y = boundary_points[i], x = boundary_points[i + 1];
        bw_img.at<cv::Vec3b>(y, x) = GREEN;
    }

    return bw_img;
}

/**
 * @brief find the boundary of a binary image using OpenCV
 *
 * @ref https://stackoverflow.com/questions/8449378/finding-contours-in-opencv
 *
 * @param img
 * @return cv::Mat*
 */
cv::Mat bw_boundry(const cv::Mat& img) {
    cv::Mat imgClone = img.clone();
    cv::cvtColor(imgClone, imgClone, cv::COLOR_BGR2GRAY);
    cv::threshold(imgClone, imgClone, 128, 255, cv::THRESH_BINARY);

    cv::Mat contourOutput = imgClone.clone();

    std::vector<std::vector<cv::Point>> contours;

    cv::findContours(contourOutput, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    cv::Mat contourImage = img.clone();
    for (size_t idx = 0ul; idx < contours.size(); ++idx) {
        cv::drawContours(contourImage, contours, idx, GREEN);
    }

    return contourImage;
}

/**
 * @brief runs the algorithm
 *
 * @param img
 * @return cv::Mat
 */
cv::Mat run_algorithm(const cv::Mat& img) {
    if (img.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        exit(EXIT_FAILURE);
    }
    const cv::Mat our_bw_boundary = radial_sweep_algorithm(img);
    const cv::Mat ocv_bw_boundary = bw_boundry(img);
    const cv::Mat combine         = one_by_three_concat(img, our_bw_boundary, ocv_bw_boundary);
    return combine;
}

/**
 * @brief Make a horizontal stack of pictures
 *
 * @param path
 * @param win_name
 */
cv::Mat one_by_three_concat(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& img3) {
    cv::Mat dst;
    cv::hconcat(img1, img2, dst);
    cv::hconcat(dst, img3, dst);
    return dst;
}

#if SHOW_GALLERY

/**
 * @brief Make a vertical stack of images
 *
 * @param frames
 * @return cv::Mat
 */
cv::Mat n_by_one_concat(const std::vector<cv::Mat> frames) {
    cv::Mat      grid;
    const size_t n = frames.size();
    cv::vconcat(frames[0], frames[1], grid);
    for (size_t i = 2; i < n; ++i) {
        cv::vconcat(grid, frames[i], grid);
    }
    return grid;
}

#endif

int main(const int argc, const char* argv[]) {
    if (argc % 2 == 0) {
        std::cerr << "Path to image or window name is not provided.\n";
        return EXIT_FAILURE;
    } else if (argc == 1) {
        std::cerr << "Please provide the path to the image.\nUsage: " << argv[0]
                  << " <path_to_image> <name of the window>\n";
        return EXIT_FAILURE;
    }

#if SHOW_GALLERY

    std::vector<cv::Mat> frame;

    for (int i = 1; i < argc; i += 2) {
        const cv::Mat org      = cv::imread(argv[i], cv::IMREAD_COLOR);
        const cv::Mat combined = run_algorithm(org);
        frame.push_back(combined);
    }
    const cv::Mat gallery = n_by_one_concat(frame);
    cv::namedWindow("GALLERY", cv::WINDOW_FREERATIO);
    cv::imshow("GALLERY", gallery);

#else

    for (int i = 1; i < argc; i += 2) {
        const cv::Mat org      = cv::imread(argv[i], cv::IMREAD_COLOR);
        const cv::Mat combined = run_algorithm(org);

        cv::namedWindow(argv[i + 1], cv::WINDOW_AUTOSIZE);
        cv::imshow(argv[i + 1], combined);
    }

#endif

    cv::waitKey(0);

    return EXIT_SUCCESS;
}