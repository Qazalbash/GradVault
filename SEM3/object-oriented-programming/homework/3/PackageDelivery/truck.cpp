#include "truck.hpp"

#include <fstream>
using namespace std;

void Truck::load(int numBox)  // loading 10 boxes in the truck
{
    for (int i = 0; i < numBox; i++) box[i];
}

void Truck::unload()  // unloading all 10 boxes from the truck
{
    ofstream tripFile;
    // writing Trip.txt file
    tripFile.open("Trip.txt", fstream::app);
    tripFile << driver << "\n";
    tripFile << petrol << "\n";
    tripFile << money << "\n";
    tripFile << fullMileage << "\n";
    tripFile << emptyMileage << "\n";

    for (int i = 0; i < 10; i++)  // inserting dimensions of box and its volume
    {
        tripFile << "Length: " << box[i].length << ", Width: " << box[i].width
                 << ", Height: " << box[i].height
                 << ", Volume: " << box[i].volume() << "\n";
    }
    tripFile.close();
}

float Truck::cost()  // calculating the cost of petrol refilled
{
    return (50 - petrol) * 2.73;
}

void Truck::update()  // updating the attributes after unloading the truck
{
    petrol = 50.0 - ((60.0 / emptyMileage) + (60.0 / fullMileage));
    money -= cost();
}

Truck::Truck(string driver_, int petrol_, int money_, int fullMileage_,
             int emptyMileage_)  // Truck constructor
{
    driver       = driver_;
    petrol       = petrol_;
    money        = money_;
    fullMileage  = fullMileage_;
    emptyMileage = emptyMileage_;
};