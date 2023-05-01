#include <cmath>
#include <iostream>

int countHoles(int num)
{
    int holes = 0, m = log10(num);
    for (int i = 0; i <= m; i++)
    {
        int digit = floor(num / pow(10, i)) - 10 * floor(num / pow(10, i + 1));
        if (digit == 0 || digit == 4 || digit == 6 || digit == 9)
            holes++;
        else if (digit == 8)
            holes += 2;
    }
    return holes;
}

int main()
{
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int num;
    std::cin >> num;
    std::cout << countHoles(num) << " holes";
    return 0;
}