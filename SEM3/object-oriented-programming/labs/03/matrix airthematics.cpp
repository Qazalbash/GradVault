#include <iostream>

int main()
{
    int n, m;
    std::cin >> n >> m;
    int v1[n][m], v2[n][m], v3[n][m];
    std::string op;

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            std::cin >> v1[i][j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            std::cin >> v2[i][j];

    std::cin >> op;

    if (op == "+")
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                v3[i][j] = v1[i][j] + v2[i][j];
    else if (op == "-")
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                v3[i][j] = v1[i][j] - v2[i][j];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            std::cout << v3[i][j] << " ";

    std::cout << std::endl;

    return 0;
}