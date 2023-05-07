#include <iostream>

struct Part
{
    int modelnumber, year, price;
};

void sort(Part parts[], const int &N)
{
    Part temp;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N - i; j++)
            if (parts[j].price > parts[j + 1].price)
            {
                temp = parts[j];
                parts[j] = parts[j + 1];
                parts[j + 1] = temp;
            }
}

void input(Part parts[], const int &N)
{
    int modelnumber, year, price;
    for (int i = 0; i < N; i++)
    {
        std::cin >> price >> modelnumber >> year;
        parts[i].price = price;
        parts[i].modelnumber = modelnumber;
        parts[i].year = year;
    }
}

void display(Part parts[], int &N)
{
    for (int k = N % 2; k < N + N % 2; k++)
    {
        std::cout << "Price: " << parts[k].price;
        std::cout << ", Part No.: " << parts[k].modelnumber;
        std::cout << ", Part Model: " << parts[k].year << std::endl;
    }
}

int main()
{
    int N;
    std::cin >> N;
    Part parts[N];

    input(parts, N);

    sort(parts, N);

    display(parts, N);

    return 0;
}