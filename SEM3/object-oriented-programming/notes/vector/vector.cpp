#include <iostream>
#include <vector>

int main()
{
        /*
        vector is a dynamic object
        std::vector<data type> name;

        when a vectrr is created cpp alocates some memory to it. when the vecotrs starts populating the memory, and
        currently allocated memory is full, cpp allocates new chuck of memory to the vector.
        */
        std::vector<int> intVector = {0, 1, 4, 7, 8, 5, 2, 3, 6, 9};

        // to add element at the last position push_back() is used
        intVector.push_back(10);

        for (int val : intVector)
                std::cout << val << " ";

        // size() returns size of the vector
        std::cout << "\nsize of vector is " << intVector.size() << std::endl;

        // front() returns value at 0 position
        std::cout << "value at front is " << intVector.front() << std::endl;

        // back() returns value at last position
        std::cout << "value at back is " << intVector.back() << std::endl;

        // at(i) returns value at ith position, it can also
        // be accesed by [] acces specifier
        std::cout << "value at 5th position is " << intVector.at(5) << std::endl;

        intVector.pop_back(); // pops value at the back

        return 0;
}