#include <iostream>

int crazySale(int nItems)
{
    int discount;

    if (nItems == 1)
        discount = 10;
    else if (nItems > 1 && nItems < 5)
        discount = 2 * nItems + 8;
    else if (nItems == 5)
        discount = 20;
    else if (nItems > 5)
        discount = 5 * (nItems - 1);

    if (discount > 70)
        discount = 70;

    return discount;
}

int main()
{
    int nItems;
    std::cin >> nItems;
    std::cout << crazySale(nItems);
    return 0;
}