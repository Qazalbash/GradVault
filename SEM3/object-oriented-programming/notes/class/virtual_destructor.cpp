#include <iostream>

class A
{
public:
        virtual ~A() { std::cout << "Destructing A" << std::endl; }
};

class B : public A
{
        int *dumyPointer;

public:
        B() { dumyPointer = new int[10]; }

        ~B()
        {
                std::cout << "Destructing B" << std::endl;
                delete dumyPointer;
        }
};

int main()
{
        A *pointerOfA;
        pointerOfA = new B();
        delete pointerOfA;

        return 0;
}