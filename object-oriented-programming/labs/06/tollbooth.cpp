#include <iostream>

class tollbooth
{
private:
    int totalCar, totalMoney;

public:
    tollbooth() : totalCar(0), totalMoney(0) {}

    void payingCar()
    {
        ++totalCar;
        totalMoney += 50;
    }

    void noPayCar() { ++totalCar; }

    void display()
    {
        std::cout << "Total cars passed: " << totalCar << std::endl;
        std::cout << "Total toll collected: Rs. " << totalMoney << std::endl;
    }
};

int main()
{
    tollbooth booth;
    char carType;

    while (carType != 'q')
    {
        std::cin >> carType;

        if (carType == 'p')
            booth.payingCar();
        else if (carType == 'n')
            booth.noPayCar();
    }

    booth.display();

    return 0;
}