#include <cstring>
#include <iostream>

int cstr_len(char *cstr)
{
    int count = 0;
    for (int i = 0; i < strlen(cstr); i++)
        count++;
    return count;
}

void append_cstring(char *dest, char *scr1, char *scr2)
{
    int length = cstr_len(scr1);
    for (int j = 0; j < length; j++)
        dest[j] = scr1[j];

    dest[length] = ' ';
    for (int j = 0; j < cstr_len(scr2); j++)
        dest[length + j + 1] = scr2[j];
}

void toggle_case(char *cstr)
{
    for (int k = 0; k < cstr_len(cstr); k++)
        cstr[k] =
            (int)cstr[k] + 32 * (((int)cstr[k] > 64 && (int)cstr[k] < 91) -
                                 ((int)cstr[k] > 96 && (int)cstr[k] < 123));
}

int main()
{
    char *scr1 = new char[100];
    char *scr2 = new char[200];
    char *dest = new char[200];

    std::cin.getline(scr1, 100);
    std::cin.getline(scr2, 100);

    std::cout << '"' << scr1 << '"' << " + ";
    std::cout << '"' << scr2 << '"' << " = ";
    append_cstring(dest, scr1, scr2);

    delete[] scr1;
    delete[] scr2;

    std::cout << '"' << dest << '"' << std::endl;

    toggle_case(dest);
    std::cout << "After case reversal: " << '"';
    std::cout << dest << '"' << std::endl;

    delete[] dest;

    return 0;
}
