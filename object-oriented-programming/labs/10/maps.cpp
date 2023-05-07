#include <iostream>
#include <map>
#include <string>

std::string sentence_without_punctuation(std::string s)
{
    std::string copy;
    int flag;

    const char punctuation_list[15] = {'.', '?', '!', ',', ';', ':', '-', '[',
                                       ']', '{', '}', '(', ')', '\'', '\"'};

    for (char wordInString : s)
    {
        flag = 0;
        for (char punctuation : punctuation_list)
        {
            if (wordInString == punctuation)
            {
                flag = 1;
                break;
            }
        }
        if (flag == 0)
            copy += wordInString;
    }

    return copy;
}

int count_vowels(std::string s)
{
    int count;
    const char vowels_list[] = {'a', 'A', 'e', 'E', 'i',
                                'I', 'o', 'O', 'u', 'U'};

    for (char wordInString : s)
        for (char vowel : vowels_list)
            if (wordInString == vowel)
                ++count;

    return count;
}

int main()
{
    std::map<std::string, std::string> map_phrases;
    std::map<std::string, int> map_vowels;
    int n;
    std::cin >> n;

    if (n < 2)
        std::cout << "Need at least 2 phrases! Better luck next time!"
                  << std::endl;
    else
    {
        int vowels;
        std::string initialSentence, sentenceWithoutPunctuation;

        for (int i = 0; i < n + 1; i++)
        {
            getline(std::cin, initialSentence);

            sentenceWithoutPunctuation =
                sentence_without_punctuation(initialSentence);

            map_phrases.insert(std::pair<std::string, std::string>(
                initialSentence, sentenceWithoutPunctuation));

            vowels = count_vowels(initialSentence);

            map_vowels.insert(
                std::pair<std::string, int>(initialSentence, vowels));
        }

        std::cout << "Displaying std::map of phrases:" << std::endl
                  << std::endl;

        for (auto mapPhraseArrow = map_phrases.begin();
             mapPhraseArrow != map_phrases.end(); mapPhraseArrow++)
        {
            if (mapPhraseArrow->first != "")
            {
                std::cout << "Original sentence: " << mapPhraseArrow->first
                          << std::endl
                          << "Without punctuation: " << mapPhraseArrow->second
                          << std::endl
                          << std::endl;
            }
        }

        std::cout << "Displaying std::map of vowels:" << std::endl
                  << std::endl;

        for (std::map<std::string, int>::iterator mapVowelArrow =
                 map_vowels.begin();
             mapVowelArrow != map_vowels.end(); mapVowelArrow++)
            if (mapVowelArrow->first != "")
                std::cout << '"' << mapVowelArrow->first << '"'
                          << " has: " << mapVowelArrow->second << " vowels"
                          << std::endl
                          << std::endl;
    }
}