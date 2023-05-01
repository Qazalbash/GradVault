#include <iostream>
#include <numeric>

struct Fraction
{
    int num, denom;
};

class FractionOperations
{
private:
    Fraction a, b;

    int gcd(int n, int m)
    {
        if (m != 0)
            return gcd(m, n % m);
        return n;
    }

public:
    FractionOperations(const Fraction aa, const Fraction bb) : a(aa), b(bb) {}

    void add()
    {
        int numerator = a.num * b.denom + a.denom * b.num;
        int denomerator = a.denom * b.denom;
        int GCD = gcd(numerator, denomerator);

        std::cout << "Result of addition is: " << numerator / GCD << "/"
                  << denomerator / GCD << std::endl;
    }

    void multiply()
    {
        int numerator = a.num * b.num;
        int denomerator = a.denom * b.denom;
        int GCD = gcd(numerator, denomerator);

        std::cout << "Result of multiplication is: " << numerator / GCD << "/"
                  << denomerator / GCD << std::endl;
    }
};

int main()
{
    Fraction a, b;
    std::cin >> a.num >> a.denom >> b.num >> b.denom;
    FractionOperations f1(a, b);
    f1.add();      // calculates a+b and prints the result
    f1.multiply(); // calculates a*b and prints the result
    return 0;
}