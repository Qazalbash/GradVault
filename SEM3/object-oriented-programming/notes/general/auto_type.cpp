#include <iostream>

/*
auto type detects the type od the argument by itself.
*/

auto sum()
{
        auto a = 0;   // a is int type
        auto b = 0.0; // b is double type
        std::cin >> a >> b;
        /*
        int + double = double
        therefore the return of the function is double.
        */
        return a + b;
}

int main()
{
        // x is a int type variable
        int x = 5;

        /*
        here y is not an auto type variable. the IDE would show the type of y when you hover the mouse pointer over it.

        auto types simply checks what type of value is been stored in the variable. in case of y, 9 is an int type which
        implies that the y is an int type variable.

        in case of z, 2.6 is a double type which implies that the z is a double type variable.

        this can be easily varified by hovering mouse pointer over each variable.
        */
        auto y = 9;
        auto z = 2.6;

        /*
        In same case auto type can be used as the return type of the funtions.

        see definiton of sum().
        */

        std::cout << sum() << std::endl;
        return 0;
}