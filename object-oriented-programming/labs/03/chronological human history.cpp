#include <iostream>

void bubble_sort(int arr[], int n)
{
    int a;
    for (int i = 0; i + 1 < n; i++)
        for (int j = 0; j + i + 1 < n; j++)
            if (arr[j] > arr[j + 1])
            {
                a = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = a;
            }
}

void result(const int arr[], const int n, const std::string corpus)
{
    std::cout << corpus;
    for (int k = 0; k + 1 < n; k++)
        std::cout << arr[k] << ", ";

    std::cout << arr[n - 1] << std::endl;
}

int main()
{
    int num_yrs, dumy;

    while (true)
    {
        std::cin >> num_yrs;
        if (num_yrs >= 2)
            break;

        std::cout << "Need at least 2 years to sort! Try Again!" << std::endl;
    }

    int years[num_yrs];

    for (int i = 0; i < num_yrs; i++)
    {
        std::cin >> dumy;
        if (dumy < 0 || dumy > 9999)
        {
            std::cout << "Year can be between 0 and 9999! Try Again!"
                      << std::endl;
            std::cin >> dumy;
        }
        years[i] = dumy;
    }

    result(years, num_yrs, "The initial array is: ");

    bubble_sort(years, num_yrs);

    result(years, num_yrs, "The sorted array is: ");

    return 0;
}