#include <iostream>
#include <string>

void race(int prime_up, int prime_down, int tron_up, int tron_down)
{
    int prime_run = 0, tron_run = 0, jumps = 0;
    std::string winner = "";

    while (true)
    {
        jumps++;
        std::cout << "Jump " << jumps << std::endl;
        prime_run += prime_up;
        tron_run += tron_up;

        if (prime_run >= 1000 && tron_run >= 1000)
        {
            std::cout << "Frog Prime has cleared the well!" << std::endl
                      << "Frogatron has cleared the well!" << std::endl;
            winner = "Tie";
            break;
        }
        else if (prime_run >= 1000)
        {
            std::cout << "Frog Prime has cleared the well!" << std::endl
                      << "Frogatron is at " << tron_run - tron_down
                      << " meters." << std::endl;
            winner = "Frog Prime";
            break;
        }
        else if (tron_run >= 1000)
        {
            std::cout << "Frog Prime is at " << prime_run - prime_down
                      << " meters." << std::endl
                      << "Frogatron has cleared the well!" << std::endl;
            winner = "Frogatron";
            break;
        }

        prime_run -= prime_down;
        tron_run -= tron_down;
        std::cout << "Frog Prime is at " << prime_run << " meters." << std::endl
                  << "Frogatron is at " << tron_run << " meters." << std::endl;
    }

    std::cout << "***** END OF RACE *****" << std::endl;
    if (winner == "Tie")
        std::cout << "TIE in " << jumps << " jumps!" << std::endl;
    else
        std::cout << winner << " wins in " << jumps << " jumps!";
}

int main(int argc, char **argv)
{
    int prime_up, prime_down, tron_up, tron_down;
    std::cin >> prime_up >> prime_down >> tron_up >> tron_down;
    race(prime_up, prime_down, tron_up, tron_down);
    return 0;
}