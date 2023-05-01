#include <iostream>

template <class T, class U>
void show(T argument1, U argument2)
{
        std::cout << "argument1 = " << argument1 << ", argument2 = " << argument2 << std::endl;
}

template <class T>
class Stack
{
        T stack[100];
        int index;

public:
        void push(T element) { stack[index++] = element; }

        T pop() { return stack[--index]; }

        Stack() : index(0) {}
};

int main()
{
        show("MEESUM", 28);

        return 0;
}