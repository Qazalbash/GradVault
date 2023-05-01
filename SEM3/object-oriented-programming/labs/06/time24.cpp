#include <iostream>

class Time24
{
    int h, m, s;
    char p;

public:
    Time24(int hour, int minute, int sec, char period = 'a')
        : h(hour), m(minute), s(sec), p(period) {}

    void show()
    {
        m += s / 60;
        s %= 60;
        h += m / 60;
        h += (p == 'p') * 12;
        m %= 60;
        h %= 24;
        if (h < 10)
            std::cout << '0';
        std::cout << h << ':';
        if (m < 10)
            std::cout << '0';

        std::cout << m << ':';
        if (s < 10)
            std::cout << '0';

        std::cout << s << std::endl;
    }

    void add(Time24 t2)
    {
        h += t2.h;
        m += t2.m;
        s += t2.s;
    }
};

using namespace std;
int main()
{
    int hours1, minutes1, seconds1;
    std::cin >> hours1 >> minutes1 >> seconds1;
    Time24 t1(hours1, minutes1, seconds1);

    int hours2, minutes2, seconds2;
    char period;
    std::cin >> hours2 >> minutes2 >> seconds2 >> period;
    Time24 t2 = {hours2, minutes2, seconds2, period};

    std::cout << "t1: ";
    t1.show();
    std::cout << "t2: ";
    t2.show();

    t1.add(t2); // result of addition is stored in t1
    std::cout << "t1+t2: ";
    t1.show();
}