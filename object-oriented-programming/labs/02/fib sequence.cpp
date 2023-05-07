#include <iostream>

void series(int term0, int term1, int n)
{
    if (n == 0)
        std::cout << term0;
    else if (n == 1)
        std::cout << term0 << ", " << term1;
    else
    {
        std::cout << term0 << ", " << term1 << ", ";
        int c;
        while (n > 1)
        {
            c = term0 + term1;
            std::cout << c;
            if (n != 2)
                std::cout << ", ";
            term0 = term1;
            term1 = c;
            n--;
        }
    }
}

int main(int argc, char **argv)
{
    int term0, term1, n;
    std::cin >> term0 >> term1 >> n;
    series(term0, term1, n);
    return 0;
}
