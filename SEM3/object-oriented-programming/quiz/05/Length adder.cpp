#include <iostream>

struct Length
{
    int feet, inches;
};

void adder(const int n)
{
    Length arr[n];

    for (int i = 0; i < n; i++)
        std::cin >> arr[i].feet >> arr[i].inches;

    int total = 0;

    for (int i = 0; i < n; i++)
        total += arr[i].feet * 12 + arr[i].inches;

    std::cout << "Sum of all lengths: " << total / 12 << "'" << total % 12
              << '"' << std::endl;
}

int main()
{
    int n;
    std::cin >> n;
    adder(n);
    return 0;
}