#include "box.cpp"
#include "truck.cpp"
#include "truckDelivery.cpp"

using namespace std;

int main()
{
    TruckDelivery delivery;   // making delivery object
    delivery.loadTrucks();    // loading the trucks
    delivery.calculateCost(); // calculating the cost of trucks
    delivery.makeJourney();   // making all trucks able to do the journey
    delivery.unloadTrucks();  // unloading all trucks that made the journey
    return 0;
}
