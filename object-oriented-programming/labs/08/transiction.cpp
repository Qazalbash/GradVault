#include <iostream>

class Payment
{
    double amount;

public:
    double getAmount() const { return this->amount; }

    void setAmount(const double &amount) { this->amount = amount; }

    void paymentDetails() const { std::cout << this->amount << std::endl; }

    Payment(const double &amount_) { this->amount = amount_; }
};

class CashPayment : Payment
{
public:
    void paymentDetails() const
    {
        std::cout << "Amount of cash payment: $";
        Payment::paymentDetails();
    }

    CashPayment(const double &amount_) : Payment(amount_){};
};

class CreditCardPayment : Payment
{
    std::string name;
    std::string expirationDate;
    std::string creditCardNumber;

public:
    void paymentDetails()
    {
        std::cout << "Amount of credit card payment: $";
        Payment::paymentDetails();
        std::cout << "Name on the credit card: " << name << std::endl
                  << "Expiration date: " << expirationDate << std::endl
                  << "Credit card number: " << creditCardNumber << std::endl;
    }

    CreditCardPayment(const double &amount_, const std::string &name_,
                      const std::string &expirationDate_,
                      const std::string &creditCardNumber_)
        : Payment(amount_),
          name(name_),
          expirationDate(expirationDate_),
          creditCardNumber(creditCardNumber_) {}
};

int main()
{
    CashPayment cp1(75.25);
    CashPayment cp2(36.95);
    CreditCardPayment ccp1(95.15, "Smith", "12/21/2009", "321654987");
    CreditCardPayment ccp2(45.75, "James", "10/30/2008", "963852741");

    std::cout << "Details of Cash #1..." << std::endl;
    cp1.paymentDetails();

    std::cout << "\nDetails of Cash #2..." << std::endl;
    cp2.paymentDetails();

    std::cout << "\nDetails of Credit Card #1..." << std::endl;
    ccp1.paymentDetails();

    std::cout << "\nDetails of Credit Card #2..." << std::endl;
    ccp2.paymentDetails();

    return 0;
}