#include "box.hpp"

using namespace std;

Box::Box() {
    // randomly generting dimensions for box
    length = 5 + rand() % 25;
    width  = 5 + rand() % 25;
    height = 5 + rand() % 25;
};

int Box::volume()  // returning volume of the box
{
    return length * height * width;
}