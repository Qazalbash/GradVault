#include <iostream>

class publication
{
    char title[50];
    float price;

public:
    void getdata() { std::cin.getline(title, 50) >> price; }

    void putdata()
    {
        std::cout << "Publication title: " << title << std::endl
                  << "Publication price: " << price << std::endl;
    }
};

class book : publication
{
    int page_count;

public:
    void getdata()
    {
        publication::getdata();
        std::cin >> page_count;
    }

    void putdata()
    {
        publication::putdata();
        std::cout << "Book page count: " << page_count << std::endl;
    }
};

class tape : publication
{
    int time;

public:
    void getdata()
    {
        publication::getdata();
        std::cin >> time;
    }

    void putdata()
    {
        publication::putdata();
        std::cout << "Tape's playing time: " << time << std::endl;
    }
};

int main()
{
    book b;
    tape t;
    b.getdata();
    std::cin.ignore();
    t.getdata();
    b.putdata();
    t.putdata();
}