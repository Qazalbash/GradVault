#include <iostream>

class TollBooth
{
    int totalCars;
    double toll;

public:
    TollBooth() : totalCars(0), toll(0) {}
    void payingCar()
    {
        ++totalCars;
        toll += 50;
    }

    void nopayCar() { totalCars++; }

    void display() const
    {
        std::cout << "Total cars passed: " << totalCars << std::endl
                  << "Total toll collected: Rs. " << toll << std::endl;
    }
};

class LyariTollBoth : public TollBooth
{
    double fine;

public:
    void trackFine()
    {
        TollBooth::nopayCar();
        fine += 500;
    }

    void display()
    {
        TollBooth::display();
        std::cout << "Total fine collected: Rs. " << fine << std::endl;
    }
};

int main()
{
    char command;
    LyariTollBoth conductorAtBooth;
    while (command != 'q')
    {
        std::cin >> command;
        if (command == 'p')
            conductorAtBooth.payingCar();
        else if (command == 'n')
            conductorAtBooth.trackFine();
    }

    conductorAtBooth.display();
    return 0;
}