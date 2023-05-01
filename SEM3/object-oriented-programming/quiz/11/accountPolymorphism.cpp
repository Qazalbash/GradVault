#include <iostream>

class Account
{
private:
    std::string title;
    double balance;

protected:
    double getBalance() { return balance; }

public:
    virtual void showDetails() { std::cout << "Title: " << title << ", "; }

    Account(const std::string title_, const double balance_)
        : title(title_), balance(balance_) {}
};

class SavingAccount : public Account
{
private:
    double ProfitRatio;

public:
    void showDetails()
    {
        Account::showDetails();
        std::cout << "Balance: " << getBalance() << ", "
                  << "Profit: " << getBalance() * ProfitRatio / 100
                  << std::endl;
    }

    SavingAccount(const std::string title_, const double balance_,
                  const double ProfitRatio_)
        : Account(title_, balance_), ProfitRatio(ProfitRatio_) {}
};

class LoanAccount : public Account
{
private:
    double IntrestRatio;

public:
    void showDetails()
    {
        Account::showDetails();
        std::cout << "Balance: " << -getBalance() << ", "
                  << "Interest: " << getBalance() * -IntrestRatio / 100
                  << std::endl;
    }

    LoanAccount(const std::string title_, const double balance_,
                const double IntrestRatio_)
        : Account(title_, balance_), IntrestRatio(IntrestRatio_) {}
};

int main()
{
    int n;
    std::cin >> n;

    std::string title;
    double balance, rate;

    for (int i = 0; i < n; i++)
    {
        std::cin >> title >> balance >> rate;
        if (balance >= 0)
        {
            SavingAccount object = {title, balance, rate};
            object.showDetails();
        }
        else
        {
            LoanAccount object = {title, balance, rate};
            object.showDetails();
        }
    }
}