class Distance
{
    int feet, inches;

public:
    Distance(int f, int i) : feet(f), inches(i) {}

    /*
    Compiler can not convert a user-defined object into basic type at its own. To convert object to any type, we
    overload that conversion type. the syntax is simple,

    operator <conversion type name>()
    {
        // code
    }

    */

    operator float() { return feet + inches / 12.0; }
};

int main()
{
    Distance d1 = {2, 3};
    Distance d2 = {7, 4};
    /*
    Distance type objects can not be directly converted to float. this ambuguity make compiler to generate error.
    */
    float length = d1;
    float width = d2;
    return 0;
}