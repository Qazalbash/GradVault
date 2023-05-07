#include "truckDelivery.hpp"

#include <fstream>
#include <iostream>
using namespace std;

void TruckDelivery::loadTrucks()  // loading the truck
{
    string   line, fDriver, lDriver, driver;
    ifstream driversFile;
    driversFile.open("Drivers.txt");
    int batchLine = 0, truckNumber = 0, petrol, money, fullMileage,
        emptyMileage;

    while (driversFile)  // reading data from Drivers.txt file
    {
        driversFile >> fDriver >> lDriver;
        driver = fDriver + " " + lDriver;
        driversFile >> petrol;
        driversFile >> money;
        driversFile >> fullMileage;
        driversFile >> emptyMileage;
        // making the object
        trucks.push_back({driver, petrol, money, fullMileage, emptyMileage});
    }
    driversFile.close();

    for (int i = 0; i < trucks.size(); i++) {
        trucks[i].load(10);  // loading the boxes
    }
}

void TruckDelivery::calculateCost()  // calculating the cost
{
    for (int i = 0; i < trucks.size(); i++) {
        trucks[i].cost();  // calulating the cost of each truck's trip
    }
}

void TruckDelivery::makeJourney()  // making each truck journey by updating th
                                   // attributes
{
    for (int i = 0; i < trucks.size(); i++) {
        trucks[i].update();  // updating the attributes after making the journey
    }
}

void TruckDelivery::unloadTrucks()  // unloading each and every truck after
                                    // their trip
{
    ofstream tripFile;
    tripFile.open("Trip.txt");  // opening trip file
    tripFile.close();
    for (int i = 0; i < trucks.size(); i++) {
        trucks[i].unload();  // calling unload function
    }
}