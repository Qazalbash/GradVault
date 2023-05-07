#include <iostream>

struct Time
{
    int hours, minutes, seconds;
};

long time_to_secs(Time t)
{
    long totalsecs = t.hours * 3600 + t.minutes * 60 + t.seconds;
    return totalsecs;
}

Time secs_to_time(long seconds)
{
    Time t;
    t.hours = seconds / 3600;
    seconds %= 3600;
    t.minutes = seconds / 60;
    seconds %= 3600;
    t.seconds = seconds;

    return t;
}

int main()
{
    std::string s;
    std::cin >> s;

    if (s == "time_to_secs")
    {
        Time t;
        std::cin >> t.hours >> t.minutes >> t.seconds;
        std::cout << time_to_secs(t) << std::endl;
    }
    else if (s == "secs_to_time")
    {
        long totalseconds;
        std::cin >> totalseconds;
        Time t = secs_to_time(totalseconds);
        std::cout << t.hours << " hours, " << t.minutes << " minutes and "
                  << t.seconds << " seconds.";
    }

    return 0;
}