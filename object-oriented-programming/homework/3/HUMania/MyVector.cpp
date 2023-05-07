// C++ program to implement Custom Vector
// class
#include <iostream>

using namespace std;

#define ll long long

// Template class to create vector of
// different data_type
template <typename DT>
class MyVector {
private:

    DT *arr;

    // Variable to store the current capacity
    // of the vector
    ll capacity;

    // Variable to store the length of the
    // vector
    ll length;

public:

    explicit MyVector(ll = 100);

    // Function that returns the number of
    // elements in array after pushing the data
    ll push_back(DT);

    // function that returns the popped element
    DT pop_back();

    // Function that return the size of vector
    ll  size() const;
    DT &operator[](ll);

    // Iterator Class
    class iterator {
    private:

        // Dynamic array using pointers
        DT *ptr;

    public:

        explicit iterator() : ptr(nullptr) {}
        explicit iterator(DT *p) : ptr(p) {}
        bool operator==(const iterator &rhs) const { return ptr == rhs.ptr; }
        bool operator!=(const iterator &rhs) const { return !(*this == rhs); }
        DT   operator*() const { return *ptr; }
        iterator &operator++() {
            ++ptr;
            return *this;
        }
        iterator operator++(int) {
            iterator temp(*this);
            ++*this;
            return temp;
        }
    };

    // Begin iterator
    iterator begin() const;

    // End iterator
    iterator end() const;
};

// Template class to return the size of
// vector of different data_type
template <typename DT>
MyVector<DT>::MyVector(ll n) : capacity(n), arr(new DT[n]), length(0) {}

// Template class to insert the element
// in vector
template <typename DT>
ll MyVector<DT>::push_back(DT data) {
    if (length == capacity) {
        DT *old = arr;
        arr     = new DT[capacity = capacity * 2];
        copy(old, old + length, arr);
        delete[] old;
    }
    arr[length++] = data;
    return length;
}

// Template class to return the popped element
// in vector
template <typename DT>
DT MyVector<DT>::pop_back() {
    return arr[length-- - 1];
}

// Template class to return the size of
// vector
template <typename DT>
ll MyVector<DT>::size() const {
    return length;
}

// Template class to return the element of
// vector at given index
template <typename DT>
DT &MyVector<DT>::operator[](ll index) {
    // if given index is greater than the
    // size of vector print Error
    if (index >= length) {
        std::cout << "Error: Array index out of bound";
        exit(0);
    }

    // else return value at that index
    return *(arr + index);
}

// Template class to return begin iterator
template <typename DT>
typename MyVector<DT>::iterator MyVector<DT>::begin() const {
    return iterator(arr);
}

// Template class to return end iterator
template <typename DT>
typename MyVector<DT>::iterator MyVector<DT>::end() const {
    return iterator(arr + length);
}

// Template class to display element in
// vector
template <typename V>
void display_vector(V &v) {
    // Declare iterator
    typename V::iterator ptr;
    for (ptr = v.begin(); ptr != v.end(); ptr++) {
        cout << *ptr << ' ';
    }
    cout << '\n';
}
