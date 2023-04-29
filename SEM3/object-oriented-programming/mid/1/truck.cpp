#include <iostream>

class Box {
private:

    int length, width, height;

public:

    Box() : length(3), width(2), height(3) {}

    int calculateVolume() { return this->length * this->width * this->height; }
};

class Truck {
private:

    std::string registrationNo;
    int         totalBoxVolume, nBoxes, maxLoadCapacity;
    bool        canMakeJourney;
    Box        *boxes;

public:

    Truck() {
        std::cin >> registrationNo >> maxLoadCapacity >> nBoxes;
        canMakeJourney = false;
        totalBoxVolume = 0;
    }

    void validate() {
        Box b;

        for (int i = 1; i <= nBoxes; i++) totalBoxVolume += b.calculateVolume();

        canMakeJourney = totalBoxVolume <= maxLoadCapacity;
    }

    friend void operator<<(std::ostream &, Truck);
};

void operator<<(std::ostream &out, Truck t) {
    out << "Truck No. " << t.registrationNo
        << " has maximum loading capacity of " << t.maxLoadCapacity
        << " m3. It carries " << t.nBoxes
        << " boxes with total dimension of boxes " << t.totalBoxVolume
        << " m3. It can ";
    if (t.canMakeJourney == false) out << "not";
    out << " make the journey successfully.\n";
}
