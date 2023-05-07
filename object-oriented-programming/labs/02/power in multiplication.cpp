#include <iostream>

int main()
{
    int base, exponent;
    std::cin >> base >> exponent;
    if (!(base >= 1 && base <= 10) || !(exponent >= 0 && exponent <= 5))
    {
        std::cout << "Invalid input!";
        return 0;
    }
    int answer = 1;
    if (exponent != 0)
        for (int i = 1; i <= exponent; i++)
            answer *= base;

    std::cout << base << " to the power of " << exponent << " is: " << answer;
    return 0;
}