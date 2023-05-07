#include <iostream>

class BankAccount
{
private:
    std::string title;
    int balance, limit, withdrawn = 0;

public:
    void deposit(int amount) { balance += amount; }

    void withdraw(int amount)
    {
        if (balance >= amount && limit - withdrawn >= amount)
        {
            balance -= amount;
            std::cout << "Withdraw Success" << std::endl;
        }
        else if (limit - withdrawn < amount && balance >= amount)
            std::cout << "Withdraw Failed: daily limit is: " << limit
                      << std::endl;
        else if (balance < amount && limit - withdrawn >= amount)
            std::cout << "Withdraw Failed: balance is insufficient."
                      << std::endl;
    }

    void ClosingStatus()
    {
        std::cout << "Closing Status:" << std::endl
                  << "Title: " << title << ", Current Balance: " << balance
                  << ", Daily limit: " << limit;
    }

    BankAccount(){};

    BankAccount(const std::string title, const int balance)
        : title(title), balance(balance)
    {
        limit = 0;
    };

    BankAccount(const std::string title, const int balance, const int limit)
        : title(title), balance(balance), limit(limit){};
};

int main()
{
    std::string title;
    int balance, limit, amount;
    char command = 0;

    std::cin >> title >> balance >> limit;

    BankAccount B = {title, balance, limit};

    while (command != 'q')
    {
        std::cin >> command >> amount;
        if (command == 'w')
            B.withdraw(amount);
        else if (command == 'd')
            B.deposit(amount);
    }

    B.ClosingStatus();

    return 0;
}
