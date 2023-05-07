#include <iostream>

int length(char str[])
{
    int count = 0;
    std::cin.get(str, 1000);
    for (int i = 0; i < 1000; i++)
    {
        if (str[i] == 0)
            return count;
        count++;
    }
    return count;
}

int main()
{
    char inputString[1000];
    std::cout << length(inputString);

    return 0;
}