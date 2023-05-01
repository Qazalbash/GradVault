#include <iostream>

/*
ennum is a user defined type for limited range of values, where values can be called from their names.
*/

enum
{
        MONDAY,    // value of MONDAY is 0
        TUESDAY,   // value of TUESDAY is 1
        WEDNESDAY, // value of WEDNESDAY is 2
        THURSDAY,  // value of THURSDAY is 3
        FRIDAY,    // value of FRIDAY is 4
        SATURDAY,  // value of SATURDAY is 5
        SUNDAY,    // value of SUNDAY is 6
};

int main()
{
        // giving each day of the week a number
        std::cout << 0 << " : Monday\n"
                  << 1 << " : Tuesday\n"
                  << 2 << " : Wednesday\n"
                  << 3 << " : Thursday\n"
                  << 4 << " : Friday\n"
                  << 5 << " : Saturday\n"
                  << 6 << " : Sunday\n";

        int choice;
        // asking user for the their choice and printing it
        std::cin >> choice;
        if (choice == 0)
                std::cout << "Monday\n";
        else if (choice == 1)
                std::cout << "Tuesday\n";
        else if (choice == 2)
                std::cout << "Wednesday\n";
        else if (choice == 3)
                std::cout << "Thursday\n";
        else if (choice == 4)
                std::cout << "Friday\n";
        else if (choice == 5)
                std::cout << "Saturday\n";
        else if (choice == 6)
                std::cout << "Sunday\n";

        // very same situation can be delat with enum data type

        std::cout << std::endl
                  << "enum situation\n\n";

        std::cout << MONDAY << " : Monday\n"
                  << TUESDAY << " : Tuesday\n"
                  << WEDNESDAY << " : Wednesday\n"
                  << THURSDAY << " : Thursday\n"
                  << FRIDAY << " : Friday\n"
                  << SATURDAY << " : Saturday\n"
                  << SUNDAY << " : Sunday\n";

        // asking user for the their choice and printing it
        std::cin >> choice;
        if (choice == MONDAY)
                std::cout << "Monday\n";
        else if (choice == TUESDAY)
                std::cout << "Tuesday\n";
        else if (choice == WEDNESDAY)
                std::cout << "Wednesday\n";
        else if (choice == THURSDAY)
                std::cout << "Thursday\n";
        else if (choice == FRIDAY)
                std::cout << "Friday\n";
        else if (choice == SATURDAY)
                std::cout << "Saturday\n";
        else if (choice == SUNDAY)
                std::cout << "Sunday\n";

        return 0;
}