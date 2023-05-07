/*
We choose composition in a situation when there is "consisits of" or "composed of" relationship among classes.

Car is consists of Doors and Engine.

It is more powerfull than aggregation. THe key point is the object classes destroyed when the main class object dies.
Like when a car is deleted, its door and engine are also deleted.
*/

class Car {
        // atributes and methods
};

class Door {
        // atributes and methods
};

class Engine {
        // atributes and methods
};