#include <iostream>

struct Complex
{
    double real, imag = 0;
};

Complex add(const Complex &z1, const Complex &z2)
{
    Complex z3;
    z3.real = z1.real + z2.real;
    z3.imag = z1.imag + z2.imag;
    return z3;
}

Complex add(const Complex &z1, const double &z2)
{
    Complex z3;
    z3.real = z1.real + z2;
    z3.imag = z1.imag;
    return z3;
}

Complex subtract(const Complex &z1, const Complex &z2)
{
    Complex z3;
    z3.real = z1.real - z2.real;
    z3.imag = z1.imag - z2.imag;
    return z3;
}

Complex subtract(const Complex &z1, const double &z2)
{
    Complex z3;
    z3.real = z1.real - z2;
    z3.imag = z1.imag;
    return z3;
}

Complex multiply(const Complex &z1, const Complex &z2)
{
    Complex z3;
    z3.real = z1.real * z2.real - z1.imag * z2.imag;
    z3.imag = z1.real * z2.imag + z1.imag * z2.real;
    return z3;
}

Complex multiply(const Complex &z1, const double &z2)
{
    Complex z3;
    z3.real = z1.real * z2;
    z3.imag = z1.imag * z2;
    return z3;
}

void show(const Complex &z)
{
    std::cout << z.real << "+";
    if (z.imag >= 0)
        std::cout << "+";
    std::cout << z.imag << "i" << std::endl;
}

int main()
{
    Complex c1, c2;
    std::cin >> c1.real >> c1.imag >> c2.real >> c2.imag;

    double d1;
    std::cin >> d1;

    std::cout << "c1+c2: ";
    show(add(c1, c2));
    std::cout << "c1-c2: ";
    show(subtract(c1, c2));
    std::cout << "c1*c2: ";
    show(multiply(c1, c2));

    std::cout << "c1+d1: ";
    show(add(c1, d1));
    std::cout << "c1-d1: ";
    show(subtract(c1, d1));
    std::cout << "c1*d1: ";
    show(multiply(c1, d1));
}