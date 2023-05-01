#include <iostream>

int steel_grade(const int hardness, const float carbon_content,
                const int tensile_strength)
{
    int H = hardness > 50;
    int C = carbon_content < 0.7f;
    int T = tensile_strength > 5600;

    if (H && C && T)
        return 10;
    else if (H && C && !T)
        return 9;
    else if (!H && C && T)
        return 8;
    else if (H && !C && T)
        return 7;
    else if ((H && !C && !T) || (!H && C && !T) || (!H && !C && T))
        return 6;
    else if (!H && !C && !T)
        return 5;

    return 0;
}

int main()
{
    int hardness, tensile_strength;
    float carbon_content;
    std::cin >> hardness >> carbon_content >> tensile_strength;
    std::cout << steel_grade(hardness, carbon_content, tensile_strength);
    return 0;
}