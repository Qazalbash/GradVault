#include <iostream>

int sum(const int &n, const int arr[], const int &a, const int &b)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
        if (arr[i] >= a && arr[i] <= b)
            sum += arr[i];
    return sum;
}

int main()
{
    int n, a, b, arr[n];

    std::cin >> n;
    for (int i = 0; i < n; i++)
        std::cin >> arr[i];

    std::cin >> a;
    std::cin >> b;

    std::cout << sum(n, arr, a, b);

    return 0;
}