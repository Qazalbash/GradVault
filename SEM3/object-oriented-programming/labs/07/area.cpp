#include <iostream>

struct Distance
{
    double feet, inch;
};

class Area
{
private:
    Distance length, width;

public:
    Area(const Distance &l, const Distance &w) : length(l), width(w) {}

    operator double()
    {
        double l = length.feet + length.inch / 12;
        double w = width.feet + width.inch / 12;
        return l * w;
    }
};

int main()
{
    double ft1, ft2, in1, in2;
    std::cin >> ft1 >> in1 >> ft2 >> in2;
    Area area1 = {
        {ft1, in1},
        {ft2, in2},
    };

    double decimalArea = area1; // overload the double operator to convert area
                                // object into decimal value.
    std::cout << "Area is: " << decimalArea << "sq feet" << std::endl;
}