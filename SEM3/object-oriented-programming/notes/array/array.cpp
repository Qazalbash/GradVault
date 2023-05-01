#include <iostream>

void increment(int B[])
{
        for (int i = 0; i < 5; i++)
        {
                B[i]++;
        }
}

int main()
{
        int A[] = {5, 2, 3, 9, 10}; // array without size

        increment(A);

        for (int i = 0; i < 5; i++)
        {
                std::cout << A[i] << std::endl;
        }

        return 0;
}