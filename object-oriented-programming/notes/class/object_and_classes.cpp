#include <iostream>

#include "LightClass.hpp"

struct LightStructure
/*
The light structure is very weak in terms of funtionality.
it is merely joining 2 variables.
*/
{
        bool state; // on or off
        int brightness;
};

/*
to modify the strucutre we are making it as a class, see LightClass.hpp
*/
LightClass::LightClass(bool st, int br)
    : state(st),
      brightness(br) // initiliizing list
/*
before going inside the body of the constructor object has been created.
actually the inside body the values are assigned.
to initialize an object we have to assign values while creating object
*/
{
        /*
        state = st;
        brightness = br;
        */
}

LightClass::LightClass(bool st) : state(st), brightness(0) {} // constructor overloading

void LightClass::turnOn() { state = true; }

void LightClass::turnOff() { state = false; }

void LightClass::showStatus()
{
        if (state)
                std::cout << "Brightness of the light is " << brightness << std::endl;
}

void LightClass::setBrightness(int n) { brightness = n; }

void LightClass::brighten()
{
        if (brightness < 10)
                brightness++;
}

void LightClass::dim()
{
        if (brightness > 0)
                brightness--;
}

int main()
{
        LightStructure LStructure = {false, 5};

        LStructure.state = true; // turning LStructure on

        // checking the brightness of the light
        if (LStructure.state)
        {
                std::cout << "Brightness of the light is " << LStructure.brightness << std::endl;
        }

        // increasing the brightness
        LStructure.brightness++;

        LStructure.state = false; // turning LStructure on

        std::cout << std::endl;

        LightClass LObject = {false, 7}; // universal form of initilization
        LightClass LObject1{true};       // = operator is optional
        LightClass LObject2(false);      // functional style of initilization

        // turning LStructure on
        LObject.turnOn();

        // setting the brightness
        LObject.setBrightness(5);

        // checking the brightness of the light
        LObject.showStatus();

        return 0;
}

/*
if you wnat only to compile the file and not to gain the execution file write the compilation command as

g++ -c <filename>.cpp

this will generate an object file <filename>.o file

if you want to use that object file to work with another file follow the command

g++ <file1name>.o <file2name>.cpp
*/