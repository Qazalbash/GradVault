#include <vector>

#include "truck.hpp"
using namespace std;

class TruckDelivery {
    // initializing the attributes
    vector<Truck> trucks;

public:

    // declearing the methods to be used in truckDelivery class
    void loadTrucks();
    void calculateCost();
    void makeJourney();
    void unloadTrucks();
};
