/*
We choose aggregation in a situation when there is "has-a" relationship among classes.

Hotel has Rooms.
Computer has CPU.
*/

class CPU
{
        // atributes and methods
};

class Computer
{
        CPU cpu; // CPU is part of the computer

        // atributes and methods
};

int main() { return 0; }