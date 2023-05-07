#include <iostream>
#include <set>
#include <string>

class Student
{
private:
    int student_id;
    std::string last_name;
    double GPA;

public:
    Student() : student_id(0), last_name(""), GPA(0.0) {}

    Student(const int &ss, const std::string &ll, const double &GG)
        : student_id(ss), last_name(ll), GPA(GG) {}

    void display() const
    {
        std::cout << this->student_id << " " << this->last_name << " "
                  << this->GPA << std::endl;
    }

    int getid() const { return this->student_id; }

    friend bool operator<(const Student &, const Student &);
    friend bool operator==(const Student &, const Student &);
};

bool operator<(const Student &s1, const Student &s2)
{
    return (s1.student_id < s2.student_id);
}

bool operator==(const Student &s1, const Student &s2)
{
    return (s1.student_id == s2.student_id);
}

int main()
{
    int id, targetId, flag = 1;
    std::string name;
    double gpa;
    std::set<Student> set_student;
    Student targetStudent;

    for (int i = 0; i < 5; i++)
    {
        std::cin >> id >> name >> gpa;

        Student student(id, name, gpa);

        set_student.insert(student);
    }

    std::cout << "Number of entries = 5" << std::endl;

    for (auto studentSetArrow : set_student)
        studentSetArrow.display();

    std::cin >> targetId;

    for (auto studentSetArrow : set_student)
    {
        if (studentSetArrow.getid() == targetId)
        {
            targetStudent = studentSetArrow;
            flag = 1;
            break;
        }
        else
            flag = 0;
    }
    if (flag)
    {
        std::cout << "Searched student is:" << std::endl;
        targetStudent.display();
    }
    else
        std::cout << "No such ID exists!";

    return 0;
}
