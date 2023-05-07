#include <iostream>
#include <map>

/*
map is a cpp structure which stores data in a key-value pair, same as dictionary in python.

map contain a pair type elements, pair contians a key an associated value to it.

map sorts the data on the basis of keys.
*/

int main()
{
        std::map<int, std::string> m;

        m[0] = "Zero"; // to insert value against some key
        m[1] = "One";
        m[2] = "Two";

        m.insert({3, "Three"}); // function to insert pair. The value must be in order

        /*
        find() is an itterator. we can also search any key value pair by [] access specifier. The problem is this when a
        key is not in the map, cpp creates the key and assign it a null value. find() does not do that.

        at(i) is also used to search key, it creates an error when key is not in the list.
        */

        if (m.find(4) != m.end())
                std::cout << "element found\n";
        else
                std::cout << "element not found\n";

        for (auto element : m)
                std::cout << "key: " << element.first << ", "
                          << "value: " << element.second << std::endl;

        return 0;
}