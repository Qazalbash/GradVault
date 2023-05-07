#include <string.h> // header for strings

#include <iostream>
#include <string> //cpp file

int main()
{
        /*
        in earlier version of c char array aws string.
        */

        char str[10] = "Hello";

        /*
        in the last of every char array, there is a null char. the ascii value of null char is 0
        */

        std::cout << str[5] << std::endl; // it will print as a blank

        for (int i = 0; str[i] != 0; i++)
        {
                std::cout << str[i] << " ";
        }
        std::cout << std::endl;

        /*
        we can also make string by assiging values to char array. when the string ends assign 0 to next element.
        */

        char str1[10];

        str1[0] = 'A';
        str1[1] = 'B';
        str1[2] = 'C';
        str1[3] = 0;

        std::cout << str1 << std::endl;

        /*
        this simple task can also be acheived by string.h file. it contains a function strcpy(str1, str2). it copies
        values of str2 to str1.
        */

        char str2[10] = "abc";
        std::cout << "str2: " << str2 << std::endl;

        std::cout << "str1 before being copied: " << str1 << std::endl;

        strcpy(str1, str2); // from string.h header file

        std::cout << "str1 after being copied: " << str1 << std::endl
                  << std::endl;

        /*
        the problem with cahr array is its limited flexibility. arrays can not be added nor they can be manipulated
        easily. Also they can not be equated to another array.

        std::string class There is no need to put size in it. string manages it by its own.
        */

        std::string s1 = "Today";    // initialization
        std::string s2("Wednesday"); // initialization
        std::string s3;
        s3 = "Sunnday"; // assignment

        if (s3 == "Sunday")
        {
                std::cout << "Its a fun day\n";
        }

        s3 = s1 + " is " + s3;

        std::cout << s3 << std::endl;

        std::cout << "length of the string is " << s3.length()
                  << std::endl; //'length() returns the length of the string

        return 0;
}