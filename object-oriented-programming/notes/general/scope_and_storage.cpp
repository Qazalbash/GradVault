#include <iostream>

/*
scope: where the function is available storage where the funciton is stored or in which class
*/

int x = 0;
/*
scope of x is availbe to entire program; global scope storage class is static: they are created here and destroyed when
the program is terminated

global values when theya re not intialized they will have 0 value. global variables are also static that means

int x = 0;
static int x = 0;

are same.

Every static variable is assigned a 0 value when it is not initialized.
*/

int main()
{
        {
                int y = 9;
                /*
                scope is only inside this function, storage is automatic: they are created here, and destroyed when this
                function {} ends

                global values when theya re not intialized they will have garbage value
                */
        }

        for (int x = 0; x < 10; x++)
        {
                static int sum = 0;
                /*
                the sum variable is created once when the loop runs for first time,

                but for other itterations of the loop sum variable is not created because its storage type is static.

                but the scope of the variable in inside the for loop, that makes it inaccesable outside the for loop

                in other words, static variables defined inside a function will not have global scope.
                */

                sum += x;

                /*
                There are 2 x is this program, one is global defined in the start of this program, other is the local
                variable used in the for loop.

                when ever the x is called inside the for loop, the locak x will be called.
                */

                std::cout << "Sum is: " << sum << std::endl;
        }

        // std::cout << sum << std::endl;
        return 0;
}