#include <iostream>

void printLine()
{
        for (int i = 0; i < 10; i++)
                std::cout << "*";
        std::cout << std::endl;
}

// most of the time the funciton is decleared in header files
void printLine1(char);

struct Distance
{
        int feet, inches;
};

int distanceInInches(Distance d)
/*
if we are declearing any type before fuunction, cpp cmpiler assumes that the return type is int.

distanceInInches(Distance d)
{
    return d.feet * 12 + d.inches;
}
*/
{
        return d.feet * 12 + d.inches;
}

void div(int x, int y, int &quotient, int &remainder)
/*
if you think the function should not return anything at all, the return type should be void.
*/
{
        quotient = x / y;
        remainder = x - quotient * y;
}

/*
The purpose of inline function is to modularize the code and also to save memory.

which means if func() is a ordinary function and it is being called the specific memory address is assesed.

in cpp inline function serves 2 purposes,

1. It serves as a compiler directive that suggests (but does not require) that the compiler substitute the body of the
function inline by performing inline expansion, i.e. by inserting the function code at the address of each function
call, thereby saving the overhead of a function call. In this respect it is analogous to the register storage class
specifier, which similarly provides an optimization hint.

2. The second purpose of inline is to change linkage behavior; the details of this are complicated. This is necessary
due to the C/C++ separate compilation + linkage model, specifically because the definition (body) of the function must
be duplicated in all translation units where it is used, to allow inlining during compiling, which, if the function has
external linkage, causes a collision during linking (it violates uniqueness of external symbols). C and C++ (and
dialects such as GNU C and Visual C++) resolve this in different ways.
*/

inline void sayHello() { std::cout << "Hello Peter\n"; }

/*
ordinary definiton of the very same function

void sayHello()
{
    std::cout << "Hello Peter\n";
}
*/

// function recurssion
int factorial(int n)
{
        if (n == 1)
                return 1;
        return n * factorial(n - 1);
}

/*
a function do some thing with integer, if i want to do the same thing with double type or long type, we do function
overloading.

function overloading can be done in 2 different ways,

1. Different type of arguments of a function
2. Different number of arguments of a function

but overloading can not be done by changing the retrun type the function

invoking 1st way to overload printMeasuremnt

void printMeasuremnt(Distance d)
void printMeasuremnt(int inches)

invoking 2nd way to overload printMeasuremnt

void printMeasuremnt(Distance d, char feetSymbol, char inchesSymbol)
*/

void printMeasuremnt(Distance d) { std::cout << "The measurment is: " << d.feet << "'" << d.inches << "\"\n"; }

void printMeasuremnt(int inches) { std::cout << "The measurment is: " << inches << std::endl; }

void printMeasuremnt(Distance d, char feetSymbol, char inchesSymbol)
{
        std::cout << "The measurment is: " << d.feet << feetSymbol << d.inches << inchesSymbol << std::endl;
}

/*
default values of a funciton are used when no argument is passed.

greetings() -> Hello
greetings("Hola") -> Hola

Function overloading can be done when function have default values as well.
*/

void greetings(std::string str = "Hello") { std::cout << str << std::endl; }

int main()
{
        printLine();
        /*
        complier will givea n error when function is defined afetr main(). to over come this error we defined the
        function before main() and write its definition after main().
        */
        printLine1('r');

        Distance d = {5, 7};
        std::cout << std::endl
                  << distanceInInches(d) << std::endl
                  << std::endl;

        int quotient, remainder;

        div(17, 5, quotient, remainder);

        std::cout << "The qoutient of 17/5 is: " << quotient << std::endl;
        std::cout << "The remiander of 17/5 is: " << remainder << std::endl
                  << std::endl;

        /*
        every time the function sayHello() is the called the control jumps to its memory location where function is
        defined.

        below there are 4 calls to function and the control will jump to function definition 4 times. this uses extra
        memory and sloes the process.

        mean while, if the function is an inline funciton the statements of the function will be copied to the point
        where it is called. apparently we can not see it, but compiler can.
        */

        int a = 2;
        sayHello();
        a = 2;
        sayHello();
        a = 5;
        sayHello();
        a = 7;
        sayHello();

        std::cout << "\n7! = " << factorial(7) << std::endl;

        return 0;
}

void printLine1(char ch)
{
        for (int i = 0; i < 10; i++)
                std::cout << ch;
        std::cout << std::endl;
}