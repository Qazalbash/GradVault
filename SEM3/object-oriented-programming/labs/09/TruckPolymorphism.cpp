#include <iostream>

class Vehicle
{
    std::string regNo;
    int wheels;

public:
    Vehicle(const std::string &regNo_, const int &wheels_)
        : regNo(regNo_), wheels(wheels_) {}

    virtual void show() const
    {
        std::cout << " Reg. No.:" << this->regNo
                  << ", No. of wheels: " << this->wheels << std::endl;
    }
};

class Truck : public Vehicle
{
    int weight;

public:
    Truck(const std::string &regNo_, const int &wheels_, const int &weight_)
        : weight(weight_), Vehicle(regNo_, wheels_) {}

    void show() const
    {
        std::cout << "This is a Truck, Max Weight: " << this->weight;
        Vehicle::show();
    }
};

int main()
{
    int n, weight, wheels;
    std::string reg;
    Vehicle *ptr;
    std::cin >> n;

    for (int i = 0; i < n; i++)
    {
        std::cin >> reg >> wheels >> weight;
        ptr = new Truck(reg, wheels, weight);

        ptr->show();

        delete ptr;
    }
}