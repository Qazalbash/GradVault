#include <iostream>

class Stack
{
        int stack[4], index;

public:
        class Range
        {
        public:
                int value;
                Range() {}
                Range(int v) : value(v) {}
        };

        void push(int element)
        {
                if (index >= 4)
                        throw Range(index);

                stack[index++] = element;
        }

        int pop()
        {
                if (index < 0)
                        throw Range(index);

                return stack[--index];
        }

        Stack() : index(0) {}
};

int main()
{
        Stack S;
        try
        {
                S.push(18);
                S.push(11);
                S.push(12);
                S.push(21);
                S.push(13);
                S.push(31);
        }
        catch (Stack::Range r)
        {
                std::cout << "Error: can not push into array. out of array bound at index " << r.value << std::endl;
        }

        try
        {
                S.pop();
                S.pop();
                S.pop();
                S.pop();
                S.pop();
                S.pop();
                S.pop();
        }
        catch (Stack::Range r)
        {
                std::cout << "Error: can not pop out of array. array is empty. out of array bound at index " << r.value
                          << std::endl;
        }
        return 0;
}