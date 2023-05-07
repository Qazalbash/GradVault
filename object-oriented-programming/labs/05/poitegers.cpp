#include <iostream>

void sort_arr(int *arr1, int sz)
{
    int dumy;
    for (int i = 0; i < sz; i++)
        for (int j = 0; j + i + 1 < sz; j++)
            if (*(arr1 + j) > *(arr1 + j + 1))
            {
                dumy = *(arr1 + j);
                *(arr1 + j) = *(arr1 + j + 1);
                *(arr1 + j + 1) = dumy;
            }
}

int binary_search(int const *arr, const int goal, int left, int right)
{
    if (right >= left)
    {
        int mid = (left + right) / 2;

        if (*(arr + mid) == goal)
            return mid;
        else if (*(arr + mid) > goal)
            return binary_search(arr, goal, left, mid - 1);
        else
            return binary_search(arr, mid + 1, right, goal);
    }
    return -1;
}

int main()
{
    int sz;
    while (true)
    {
        std::cin >> sz;
        if (sz < 1)
            std::cout << "Incorrect size! Try Again" << std::endl;
        else
            break;
    }
    int *arr1 = new int[sz];
    for (int k = 0; k < sz; k++)
        std::cin >> *(arr1 + k);

    std::cout << "Before sorting: ";
    for (int k = 0; k < sz - 1; k++)
        std::cout << *(arr1 + k) << ", ";

    std::cout << *(arr1 + sz - 1) << std::endl;

    sort_arr(arr1, sz);

    std::cout << "After sorting: ";
    for (int k = 0; k < sz - 1; k++)
        std::cout << *(arr1 + k) << ", ";

    std::cout << *(arr1 + sz - 1) << std::endl;

    int goal;
    std::cin >> goal;
    std::cout << std::endl
              << "The value to be searched: " << goal << std::endl;
    int search = binary_search(arr1, goal, 0, sz - 1);
    if (search != -1)
        std::cout << "Element is present at index: " << search << std::endl;
    else
        std::cout << "Element is not present in the array!";

    return 0;
}