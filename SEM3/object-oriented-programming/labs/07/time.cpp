#include <iostream>

class Time
{
    int h, m, s;

public:
    Time(const int &hour, const int &minute, const int &sec)
        : h(hour), m(minute), s(sec) {}

    Time() : h(0), m(0), s(0) {}

    void show()
    {
        m += s / 60;
        s %= 60;
        h += m / 60;
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

    Time operator+(const Time &t2)
    {
        Time t3;
        t3.h = h + t2.h;
        t3.m = m + t2.m;
        t3.s = s + t2.s;
        return t3;
    }
};

int main()
{
    int hh, mm, ss;
    std::cin >> hh >> mm >> ss;
    Time t1(hh, mm, ss); // t1(hh, mm, ss)
    std::cin >> hh >> mm >> ss;
    Time t2(hh, mm, ss);
    Time t3;
    t3 = t1 + t2;

    t1.show();
    t2.show();
    t3.show();
    return 0;
}