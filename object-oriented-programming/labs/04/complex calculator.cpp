#include <iostream>

struct Complex
{
    double real, imag;
};

Complex add(const Complex &z1, const Complex &z2, int direction)
{
    Complex z3;
    z3.real = z1.real + direction * z2.real;
    z3.imag = z1.imag + direction * z2.imag;
    return z3;
}

Complex multiply(const Complex &z1, const Complex &z2)
{
    Complex z3;
    z3.real = z1.real * z2.real - z1.imag * z2.imag;
    z3.imag = z1.real * z2.imag + z1.imag * z2.real;
    return z3;
}

void show(const Complex z)
{
    std::cout << z.real;
    if (z.imag >= 0)
        std::cout << "+";
    std::cout << z.imag << "i" << std::endl;
}

int main()
{
    Complex z1, z2;
    std::cin >> z1.real >> z1.imag >> z2.real >> z2.imag;

    std::cout << "Addition: ";
    show(add(z1, z2, 1));
    std::cout << "Subtraction: ";
    show(add(z1, z2, -1));
    std::cout << "Multiplication: ";
    show(multiply(z1, z2));

    return 0;
}