#include <iostream>

class Complex
{
private:
    double real, imag;

public:
    void show()
    {
        std::cout << this->real;
        if (this->imag >= 0)
            std::cout << "+";
        std::cout << this->imag << "i" << std::endl;
    }

    Complex add(const Complex &z)
    {
        return Complex{this->real + z.real, this->imag + z.imag};
    }

    Complex subtract(Complex z)
    {
        return Complex{this->real - z.real, this->imag - z.imag};
    }

    Complex multiply(Complex z)
    {
        return Complex{this->real * z.real - this->imag * z.imag,
                       this->real * z.imag + this->imag * z.real};
    }

    Complex add(double z) { return Complex{this->real + z, this->imag}; }

    Complex subtract(double z) { return Complex{this->real - z, this->imag}; }

    Complex multiply(double z) { Complex{this->real * z, this->imag * z}; }

    Complex() : real(0), imag(0) {}

    Complex(double r, double i) : real(r), imag(i) {}
};

int main()
{
    double real, imag;
    std::cin >> real >> imag;
    Complex c1 = {real, imag};

    std::cin >> real >> imag;
    Complex c2 = {real, imag};

    double d1;
    std::cin >> d1;

    Complex result;
    // showing the numbers:
    std::cout << "c1: ";
    c1.show();
    std::cout << "c2: ";
    c2.show();
    std::cout << "d1: " << d1 << std::endl;

    // Check the opertions where both operands are complex
    result = c1.add(c2);
    std::cout << "c1+c2: ";
    result.show();

    result = c1.subtract(c2);
    std::cout << "c1-c2: ";
    result.show();

    result = c1.multiply(c2);
    std::cout << "c1*c2: ";
    result.show();

    // Check the opertions where one operator is complex, other is double

    result = c1.add(d1);
    std::cout << "c1+d1: ";
    result.show();

    result = c1.subtract(d1);
    std::cout << "c1-d1: ";
    result.show();

    result = c1.multiply(d1);
    std::cout << "c1*d1: ";
    result.show();
}