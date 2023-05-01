#include <iostream>

class Complex
{
private:
    double real, imag;

public:
    Complex operator+(const Complex &z)
    {
        Complex z2 = {this->real + z.real, this->imag + z.imag};
        return z2;
    }

    Complex operator-(const Complex &z)
    {
        Complex z2 = {this->real - z.real, this->imag - z.imag};
        return z2;
    }

    Complex operator*(const Complex &z)
    {
        Complex z2 = {this->real * z.real - this->imag * z.imag,
                      this->real * z.imag + this->imag * z.real};
        return z2;
    }

    Complex operator+(const double &z)
    {
        Complex z2 = {this->real + z, this->imag};
        return z2;
    }

    Complex operator-(const double &z)
    {
        Complex z2 = {this->real - z, this->imag};
        return z2;
    }

    Complex operator*(const double &z)
    {
        Complex z2 = {this->real * z, this->imag * z};
        return z2;
    }

    Complex() : real(0), imag(0) {}

    Complex(const double &real, const double &imag) : real(real), imag(imag) {}

    friend std::istream &operator>>(std::istream &, Complex &);
    friend std::ostream &operator<<(std::ostream &, const Complex &);
};

std::istream &operator>>(std::istream &in, Complex &c)
{
    in >> c.real;
    in >> c.imag;
    return in;
}

std::ostream &operator<<(std::ostream &out, const Complex &c)
{
    out << c.real;
    if (c.imag >= 0)
        out << "+";
    out << c.imag << "i";
    return out;
}

int main()
{
    Complex c1, c2;

    std::cin >> c1; // extraction operator is overloaded
    std::cin >> c2;

    double d1;
    std::cin >> d1;

    Complex result;
    // showing the numbers:
    std::cout << "c1: " << c1 << std::endl; // insertion operator is overloaded
    std::cout << "c2: " << c2 << std::endl;
    std::cout << "d1: " << d1 << std::endl;

    // Check the opertions where both operands are complex
    result = c1 + c2;
    std::cout << "c1+c2: " << result << std::endl;

    result = c1 - c2;
    std::cout << "c1-c2: " << result << std::endl;

    result = c1 * c2;
    std::cout << "c1*c2: " << result << std::endl;

    // Check the opertions where one operator is complex, other is double

    result = c1 + d1;
    std::cout << "c1+d1: " << result << std::endl;

    result = c1 - d1;
    std::cout << "c1-d1: " << result << std::endl;

    result = c1 * d1;
    std::cout << "c1*d1: " << result << std::endl;
}