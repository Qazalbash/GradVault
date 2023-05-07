#include <iostream>
#include <string>

void intern(std::string input, std::string output)
{
    for (int i = 0; i < input.length(); i++)
    {
        if (input[i] == 'a')
            output += "A";
        else if (input[i] == 'c')
            output += "C";
        else if (input[i] == 'g')
            output += "G";
        else if (input[i] == 't')
            output += "T";
        else if (input[i] == 'A' || input[i] == 'C' || input[i] == 'G' ||
                 input[i] == 'T')
            output += input[i];
    }
    std::cout << output;
}

int main()
{
    std::string input, output = "";
    getline(std::cin, input);
    intern(input, output);

    return 0;
}