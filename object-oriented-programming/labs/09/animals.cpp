#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

class Animal
{
protected:
    std::string name, sound;

public:
    virtual void make_sound() const = 0;

    Animal(const std::string &name_, const std::string &sound_)
        : name(name_), sound(sound_) {}
};

class Cat : public Animal
{
    std::string action;

public:
    Cat(const std::string &name_, const std::string &sound_ = "Meow",
        const std::string &action_ = "purrs")
        : Animal(name_, sound_), action(action_) {}

    void make_sound() const
    {
        std::cout << Animal::name << " " << action << ": " << Animal::sound
                  << "!" << std::endl;
    }
};

class Cow : public Animal
{
    std::string action;

public:
    Cow(const std::string &name_, const std::string &sound_ = "Moo",
        const std::string &action_ = "moos")
        : Animal(name_, sound_), action(action_) {}

    void make_sound() const
    {
        std::cout << Animal::name << " " << action << ": " << Animal::sound
                  << "!" << std::endl;
    }
};

class Dog : public Animal
{
    std::string action;

public:
    Dog(const std::string &name_, const std::string &sound_ = "Woof",
        const std::string &action_ = "barks")
        : Animal(name_, sound_), action(action_) {}

    void make_sound() const
    {
        std::cout << Animal::name << " " << action << ": " << Animal::sound
                  << "!" << std::endl;
    }
};

class Duck : public Animal
{
    std::string action;

public:
    Duck(const std::string &name_, const std::string &sound_ = "Quack",
         const std::string &action_ = "quacks")
        : Animal(name_, sound_), action(action_) {}

    void make_sound() const
    {
        std::cout << Animal::name << " " << action << ": " << Animal::sound
                  << "!" << std::endl;
    }
};

class Horse : public Animal
{
    std::string action;

public:
    Horse(const std::string &name_, const std::string &sound_ = "Neigh",
          const std::string &action_ = "nickers")
        : Animal(name_, sound_), action(action_) {}

    void make_sound() const
    {
        std::cout << Animal::name << " " << action << ": " << Animal::sound
                  << "!" << std::endl;
    }
};

class Pig : public Animal
{
    std::string action;

public:
    Pig(const std::string &name_, const std::string &sound_ = "Oink",
        const std::string &action_ = "snorts")
        : Animal(name_, sound_), action(action_) {}

    void make_sound() const
    {
        std::cout << Animal::name << " " << action << ": " << Animal::sound
                  << "!" << std::endl;
    }
};

void split(const std::string &str, std::vector<std::string> &v)
{
    std::stringstream ss(str);
    ss >> std::noskipws;
    std::string field;
    char ws_delim;
    while (1)
    {
        if (ss >> field)
            v.push_back(field);
        else if (ss.eof())
            break;
        else
            v.push_back(std::string());

        ss.clear();
        ss >> ws_delim;
    }
}

void caseBreaker(Animal **animal, int number, int animalType, std::string name,
                 std::string sound = "")
{
    if (sound != "")
    {
        switch (animalType)
        {
        case 1:
            animal[number] = new Cat(name, sound);
            break;
        case 2:
            animal[number] = new Cow(name, sound);
            break;
        case 3:
            animal[number] = new Dog(name, sound);
            break;
        case 4:
            animal[number] = new Duck(name, sound);
            break;
        case 5:
            animal[number] = new Horse(name, sound);
            break;
        case 6:
            animal[number] = new Pig(name, sound);
            break;
        }
    }
    else
    {
        switch (animalType)
        {
        case 1:
            animal[number] = new Cat(name);
            break;
        case 2:
            animal[number] = new Cow(name);
            break;
        case 3:
            animal[number] = new Dog(name);
            break;
        case 4:
            animal[number] = new Duck(name);
            break;
        case 5:
            animal[number] = new Horse(name);
            break;
        case 6:
            animal[number] = new Pig(name);
            break;
        }
    }
}

int main()
{
    int noOfAnimals, animalType;

    std::vector<std::vector<std::string>> dataVector;

    std::string inputString, name, sound;

    std::cin >> noOfAnimals;
    std::cin.ignore();

    Animal **animals = new Animal *[noOfAnimals];

    for (int i = 0; i <= noOfAnimals + 1; i++)
    {
        std::vector<std::string> subVector = {};

        getline(std::cin, inputString);
        split(inputString, subVector);

        dataVector.push_back(subVector);
    }

    for (int i = 0; i < noOfAnimals; i++)
    {
        if (dataVector[i].size() == 2)
            caseBreaker(animals, i, stoi((dataVector[i])[0]),
                        (dataVector[i])[1]);
        else if (dataVector[i].size() == 3)
            caseBreaker(animals, i, stoi((dataVector[i])[0]),
                        (dataVector[i])[1], (dataVector[i])[2]);
    }

    for (int k = 0; k < (dataVector[noOfAnimals]).size(); k++)
        animals[stoi(dataVector[noOfAnimals][k]) - 1]->make_sound();

    dataVector.clear();
    return 0;
}