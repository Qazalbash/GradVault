#include <iostream>

int *count = new int(0);

void callMe()
{
    *count = *count + 1;
    std::cout << "I have been called " << *count << " times" << std::endl;
    ;
}

int main()
{
    for (int i = 0; i < 10; i++)
        callMe();

    delete count;
    return 0;
}