#include <iostream>

struct point
{
    int x, y, z;
};

point add(const point &p1, const point &p2, const int &direction)
{
    point p3;
    p3.x = p1.x + direction * p2.x;
    p3.y = p1.y + direction * p2.y;
    p3.z = p1.z + direction * p2.z;
    return p3;
}

point prod(const point &p1, const point &p2)
{
    point p3;
    p3.x = p1.x * p2.x;
    p3.y = p1.y * p2.y;
    p3.z = p1.z * p2.z;
    return p3;
}

void display(const std::string &corpus, const point &p)
{
    std::cout << corpus;
    std::cout << "(" << p.x << ", " << p.y << ", " << p.z << ")" << std::endl;
}

int main()
{
    point p1, p2;

    std::cin >> p1.x >> p1.y >> p1.z;
    std::cin >> p2.x >> p2.y >> p2.z;
    display("Coordinates of p1 are: ", p1);
    display("Coordinates of p2 are: ", p2);
    display("Coordinates of p1 + p2 are: ", add(p1, p2, 1));
    display("Coordinates of p1 - p2 are: ", add(p1, p2, -1));
    display("Coordinates of p1 * p2 are: ", prod(p1, p2));

    return 0;
}