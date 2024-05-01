#include <iostream>
#include <opencv2/opencv.hpp>

#define SHOW_IMAGE false

#if (SHOW_IMAGE)

void show_image(const std::string& win_name, const cv::Mat& img) {
    cv::namedWindow(win_name, cv::WINDOW_AUTOSIZE);
    cv::imshow(win_name, img);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

#endif

/**
 * @brief Demosaics a Bayer filter image using the BGGR pattern to produce a color image.
 *
 * @param img (cv::Mat): Input image in the BGGR Bayer filter pattern.
 *
 * @return cv::Mat: Demosaiced color image with three color channels (BGR).
 *
 * @details This function takes an input image in the BGGR Bayer filter pattern and processes it to produce a demosaiced
 * color image. It iterates through each pixel of the input image, determining the pixel type (Blue, Green, or Red)
 * based on the pixel's parity and boundary status. Then, it calculates the blue, green, and red color channel values
 * for each pixel and assigns them to the output image. The resulting image is a color image with three channels (BGR).
 *
 * @example
 *      cv::Mat img = cv::imread('mandi.tif')
 *      cv::Mat color_img = bayer_filter_BGGR(img)
 *      cv::imwrite('demosaiced_mandi.tif', color_img)
 */
cv::Mat bayer_filter_BGGR(const cv::Mat& img) {
    const size_t rows = img.rows, cols = img.cols;

    // Create an output image to store the demosaiced color image.
    cv::Mat demosaiced_img = cv::Mat::zeros(img.size(), CV_8UC3);
    int     b, g, r;  // Variables to store color channel values.

    // Iterate through each pixel in the input image.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // Determine pixel parity, which helps identify its type (Blue, Green, or Red).
            // parity = 0b(i%2)(j%2)
            const uchar parity = ((i & 0b1) << 1) | (j & 0b1);

            // Determine pixel boundary status to handle edge cases.
            // bound = 0b(0<i)(i<rows-1)(0<j)(j<cols-1)
            const uchar bound = ((0 < i) << 3) | ((i < rows - 1) << 2) | ((0 < j) << 1) | (j < cols - 1);

            // Switch statement to handle different pixel types based on parity.
            switch (parity) {
                case 0b00:  // Blue pixel
                            // Calculate the blue, green, and red values for this pixel.
                    b = img.at<uchar>(i, j);
                    g = (bound & 0b1) ? ((img.at<uchar>(i, j + 1) + img.at<uchar>(i + 1, j)) >> 1) : 0;
                    r = ((bound & 0b101) == 0b101) ? img.at<uchar>(i + 1, j + 1) : 0;
                    break;

                case 0b01:  // Green pixel
                            // Calculate the blue, green, and red values for this pixel.
                    b = (bound & 0b1) ? img.at<uchar>(i, j + 1) : 0;
                    g = img.at<uchar>(i, j);
                    r = (bound & 0b100) ? img.at<uchar>(i + 1, j) : 0;
                    break;

                case 0b10:  // Green pixel
                            // Calculate the blue, green, and red values for this pixel.
                    b = (bound & 0b100) ? img.at<uchar>(i + 1, j) : 0;
                    g = img.at<uchar>(i, j);
                    r = (bound & 0b1) ? img.at<uchar>(i, j + 1) : 0;
                    break;

                default:  // Red pixel
                          // Calculate the blue, green, and red values for this pixel.
                    b = ((bound & 0b1010) == 0b1010) ? img.at<uchar>(i - 1, j - 1) : 0;
                    g = ((bound & 0b110) == 0b110) ? ((img.at<uchar>(i, j - 1) + img.at<uchar>(i + 1, j)) >> 1) : 0;
                    r = img.at<uchar>(i, j);
                    break;
            }
            // Assign the calculated color values to the output image.
            demosaiced_img.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
        }
    }
    // Return the demosaiced color image.
    return demosaiced_img;
}

int main(const int argc, const char* argv[]) {
    // Check if the program was provided with the correct number of command-line arguments (2 arguments expected).
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input-image-path> <output-image-path>\n";
        // Print an error message to the standard error stream and explain how to use the program.
        return EXIT_FAILURE;  // Exit the program with a failure status code.
    }

    // Load the input image specified by the first command-line argument and convert it to grayscale.
    const cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        // Check if the input image could not be loaded or does not exist.
        std::cerr << "Image not found\n";
        // Print an error message to the standard error stream.
        return EXIT_FAILURE;  // Exit the program with a failure status code.
    }

    // Demosaic the grayscale input image using the "bayer_filter_BGGR" function to create a color image.
    const cv::Mat demosaiced_img = bayer_filter_BGGR(img);

#if (SHOW_IMAGE)

    // Show the resulting demosaiced color image.
    show_image("Demosaiced Image", demosaiced_img);

#else

    // Save the resulting demosaiced color image to the path specified by the second command-line argument.
    cv::imwrite(argv[2], demosaiced_img);
    std::cout << "Image has been saved as " << argv[2] << "\n";

#endif

    return EXIT_SUCCESS;  // Exit the program with a success status code.
}