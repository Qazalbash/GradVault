#include "Image.hpp"

/**
 * @brief Construct a new Image:: Image object
 *
 * @param h
 * @param v
 */
Image::Image(int h, int v) {
    // Assign hres and vrex from parameters
    this->hres = h;
    this->vres = v;

    // Assign color pointer arrary
    this->colors = new RGBColor*[v];
    // Assign color arrary
    for (int i = 0; i < v; i++) colors[i] = new RGBColor[h];
}

/**
 * @brief Construct a new Image:: Image object
 *
 * @param vp
 */
Image::Image(const ViewPlane& vp) {
    // Assign hres and vrex from Viewplane
    this->hres = vp.get_hres();
    this->vres = vp.get_vres();

    // Assign color pointer arrary
    this->colors = new RGBColor*[this->vres];
    // Assign color arrary
    for (int i = 0; i < (this->vres); i++) colors[i] = new RGBColor[this->hres];
}

/**
 * @brief Destroy the Image:: Image object
 *
 */
Image::~Image() {
    // Delete pointers of each RGBcolor array
    for (int i = 0; i < this->vres; i++) delete[] this->colors[i];
    // Delete color**
    delete[] this->colors;
}

/**
 * @brief Set the pixel object at x,y to color
 *
 * @param x
 * @param y
 * @param color
 */
void Image::set_pixel(int x, int y, const RGBColor& color) {
    this->colors[x][y] = color;
}

void Image::set_pixel(int x, int y, const RGBColor& color, int samples) {
    this->colors[x][y].r = color.r / samples;
    this->colors[x][y].g = color.g / samples;
    this->colors[x][y].b = color.b / samples;
}

/**
 * @brief Write the image to a ppm file
 *
 * @param path
 */
void Image::write_ppm(std::string path) const {
    float maxRGB = 0, temp;
    int   i = 0, j = 0;

    for (; i < vres; i++) {
        for (j = 0; j < hres; j++) {
            // Find largest value in all the r,g,b values in the color array.
            temp = max_element(colors[i][j].r, colors[i][j].g, colors[i][j].b);
            if (temp > maxRGB) maxRGB = temp;
        }
    }
    // Find scale using maxRGB
    double scale = 255 / maxRGB;

    std::ofstream file(path);

    file << "P3\n";
    file << hres << " " << vres << "\n";
    file << "255\n";

    for (i = 0; i < vres; i++) {
        for (j = 0; j < hres; j++) {
            file << std::to_string(colors[j][i].r * scale) << " "
                 << std::to_string(colors[j][i].g * scale) << " "
                 << std::to_string(colors[j][i].b * scale) << " ";
        }
        file << "\n";
    }
    file.close();
}
