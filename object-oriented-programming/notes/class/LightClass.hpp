class LightClass {
        /*
        Normally the attributes are set private, and functions are set public.

        The LightClass is more useable and flexiable than LightStructure, because user doesn't have to write its own
        function when ever using.
        */

    private:

        bool state;
        int  brightness;

    public:

        /*
        to initialize objects a special type of function is called, named as constructor

        Constructor has same name has class name it does not have any return type it is called automatically when an
        object is created

        we have defined 2 constructor in the class, the ddefault constructor of cpp is no more available here.
        */
        LightClass(bool st, int br) : state(st), brightness(br) {}
        LightClass(bool st) : state(st), brightness(0) {}  // constructor overloading
        void turnOn();
        void turnOff();
        void showStatus();
        void setBrightness(int n);
        void brighten();
        void dim();
};