#include <iostream>

/*
A structure is a collection of simple variables. The variables in a structure can be of different types. The data items
in a structure are called the members of the structure.

No memory has been utilized, because no variable has been created.
*/

struct Distance
{
        int feet;
        int inches;
};

struct Rectangle
{
        Distance length;
        Distance width;
};

void adjust(Distance dist[], int N)
{
        for (int i = 0; i < N; i++)
        {
                if (dist[i].inches > 11)
                {
                        dist[i].feet += dist[i].inches / 12;
                        dist[i].inches %= 12;
                }
        }
}

int main()
{
        Distance d1;   // creating distance type variable
        d1.feet = 6;   // assigning value to feets
        d1.inches = 1; // assigning value to inches

        std::cout << "The distance of d1: " << d1.feet << "'" << d1.inches << "\"\n\n";

        Distance d2 = {5, 7}; /* d2 is initialized in order by which
                               they are crated inside the structure */
        std::cout << "The distance of d2: " << d2.feet << "'" << d2.inches << "\"\n\n";

        d2 = {4, 11}; // structure overwriting
        std::cout << "The distance of d2 after overwriting: " << d2.feet << "'" << d2.inches << "\"\n\n";

        d1 = d2; // copying d2 to d1
        std::cout << "The distance of d1 after copying: " << d2.feet << "'" << d2.inches << "\"\n\n";

        Rectangle R = {
            {2, 3},
            {3, 5}}; // initializing rectangle type variable
        std::cout << "Rectangle length: " << R.length.feet << "'" << R.length.inches << "\"\n";
        std::cout << "Rectangle width: " << R.width.feet << "'" << R.width.inches << "\"\n\n";

        Distance dist[5]; // Distance type array
        // every member of this array is a Distance type variable
        for (int i = 0; i < 5; i++)
                dist[i] = {3 * i, i + 1};

        for (int i = 0; i < 5; i++)
                std::cout << dist[i].feet << "'" << dist[i].inches << "\"\n";

        std::cout << std::endl;

        Distance dist1[5];
        for (int i = 0; i < 5; i++)
        {
                // another valid assignment
                dist1[i].feet = 3 * i;
                dist1[i].inches = i + 1;
        }

        for (int i = 0; i < 5; i++)
                std::cout << dist1[i].feet << "'" << dist1[i].inches << "\"\n";
        std::cout << std::endl;

        // another valid initialization
        Distance dist2[5] = {
            {1, 2},
            {8, 15},
            {4, 1},
            {6, 12},
            {5, 13},
        };

        adjust(dist2, 5); // caliing a function with Ditance type array as an input

        for (int i = 0; i < 5; i++)
                std::cout << dist2[i].feet << "'" << dist2[i].inches << "\"\n";

        return 0;
}