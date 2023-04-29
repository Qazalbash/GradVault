#include <iostream>
#include <vector>

std::vector<int> find_local_minimums(const std::vector<int>& v) {
    std::vector<int> local_minimums;

    size_t left  = 0UL;
    size_t right = v.size() - 1;

    if (v[0] <= v[1]) local_minimums.push_back(v[0]);
    if (v[right] <= v[right - 1]) local_minimums.push_back(v[right]);

    while (left < right) {
        int mid = (left + right) >> 1;

        if (v[mid] <= v[mid - 1] && v[mid] <= v[mid + 1]) local_minimums.push_back(v[mid]);

        if (v[mid] <= v[mid - 1] && v[mid] <= v[mid + 1])
            left = mid + 1;
        else if (v[mid] >= v[mid - 1] && v[mid] >= v[mid + 1])
            right = mid - 1;
        else if (v[mid] <= v[mid - 1] && v[mid] >= v[mid + 1])
            left = mid + 1;
        else if (v[mid] >= v[mid - 1] && v[mid] <= v[mid + 1])
            left = mid + 1;
        else if (v[mid] == v[mid - 1] && v[mid] <= v[mid + 1])
            left = mid + 1;
        else if (v[mid] == v[mid - 1] && v[mid] >= v[mid + 1])
            right = mid - 1;
    }

    return local_minimums;
}