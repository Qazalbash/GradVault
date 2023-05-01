#include <iostream>

void vectorInput(int vector[], const int n)
{
    for (int i = 0; i < n; i++)
        std::cin >> vector[i];
}

void vector(const int n, int v1[], int v2[], const std::string op, int v3[])
{
    if (op == "+")
        for (int i = 0; i < n; i++)
            v3[i] = v1[i] + v2[i];
    else if (op == "-")
        for (int i = 0; i < n; i++)
            v3[i] = v1[i] - v2[i];
}

void comparision(const int v1[], const int v2[], const int n)
{
    int flag = 0;
    for (int i = 0; i < n; i++)
        if (v1[i] != v2[i])
        {
            std::cout << "UNEQUAL";
            flag = 1;
            break;
        }
    if (flag == 0)
        std::cout << "EQUAL";
}

int main()
{
    int n;
    std::cin >> n;
    int v1[n], v2[n];
    std::string op;

    vectorInput(v1, n);
    vectorInput(v2, n);

    std::cin >> op;
    if (op == "+" || op == "-")
    {
        int v3[n];
        vector(n, v1, v2, op, v3);
        for (int i = 0; i < n; i++)
            std::cout << v3[i] << " ";
    }
    else
        comparision(v1, v2, n);

    return 0;
}