#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define ALLOWED_CARS 3 /* Number of cars allowed on street at a time */
#define USAGE_LIMIT 7  /* Number of times street can be used before repair */
#define MAX_CARS 1000  /* Maximum number of cars in the simulation */

#define INCOMING "Incoming"
#define OUTGOING "Outgoing"

/* synchronization variables */

pthread_mutex_t mutex; // mutex lock for the critical section of the code
pthread_cond_t
    cond; // condition variable for incoming cars waiting to enter the street

/* These obvious variables are at your disposal. Feel free to remove them if you
 * want */
static int cars_on_street;    // Total number of cars currently on street
static int incoming_onstreet; // Total number of cars incoming on street
static int outgoing_onstreet; // Total number of cars outgoing on street
static int cars_since_repair; // Total number of cars entered since last repair

typedef struct
{
    int arrival_time; // time between the arrival of this car and the previous
                      // car
    int travel_time;  // time the car takes to travel on the street
    int car_id;
    char car_direction[20];
} car;

/* Called at the starting of simulation. Initialize all synchronization
 * variables and other global variables that you add.
 */
static int initialize(car *arr, char *filename)
{
    cars_on_street = 0;
    incoming_onstreet = 0;
    outgoing_onstreet = 0;
    cars_since_repair = 0;

    pthread_mutex_init(&mutex, NULL); // initialize the mutex lock
    pthread_cond_init(&cond, NULL);   // initialize the condition variable

    /* Read in the data file and initialize the car array */
    FILE *fp;

    if ((fp = fopen(filename, "r")) == NULL)
    {
        printf("Cannot open input file %s for reading.\n", filename);
        exit(1);
    }
    int i = 0;
    while ((fscanf(fp, "%d%d%s\n", &(arr[i].arrival_time),
                   &(arr[i].travel_time), arr[i].car_direction) != EOF) &&
           i < MAX_CARS)
        i++;
    fclose(fp);
    return i;
}

/* Code executed by street on the event of repair
 * Do not add anything here.
 */
static void repair_street()
{
    printf("The street is being repaired now.\n");
    sleep(5);
}

/***********************FUNCTIONS*OF*INTEREST***********************/

void *street_thread(void *junk)
{
    printf("The street is ready to use\n");
    while (1)
    {
        if (cars_since_repair == USAGE_LIMIT)
        {
            pthread_mutex_lock(&mutex);   // lock the mutex
            cars_since_repair = 0;        // reset the counter
            repair_street();              // call the repair function
            pthread_cond_signal(&cond);   // signal the condition variable
            pthread_mutex_unlock(&mutex); // unlock the mutex
        }
    }
    pthread_exit(NULL); // exit the thread
}

void incoming_enter()
{
    pthread_mutex_lock(&mutex); // lock the mutex
    while (cars_on_street >= ALLOWED_CARS || cars_since_repair >= USAGE_LIMIT ||
           outgoing_onstreet >
               0)                         // wait until the street is ready to accept incoming cars
        pthread_cond_wait(&cond, &mutex); // wait on the condition variable
    cars_on_street++;                     // increment the number of cars on the street
    incoming_onstreet++;                  // increment the number of incoming cars on the street
    pthread_mutex_unlock(&mutex);         // unlock the mutex
}

void outgoing_enter()
{
    pthread_mutex_lock(&mutex); // lock the mutex
    while (cars_on_street >= ALLOWED_CARS || cars_since_repair >= USAGE_LIMIT ||
           incoming_onstreet >
               0)                         // wait until the street is ready to accept outgoing cars
        pthread_cond_wait(&cond, &mutex); // wait on the condition variable
    cars_on_street++;                     // increment the number of cars on the street
    outgoing_onstreet++;                  // increment the number of outgoing cars on the street
    pthread_mutex_unlock(&mutex);         // unlock the mutex
}

static void travel(int t) { sleep(t); }

static void incoming_leave()
{
    pthread_mutex_lock(&mutex);   // lock the mutex
    cars_on_street--;             // decrement the number of cars on the street
    incoming_onstreet--;          // decrement the number of incoming cars on the street
    cars_since_repair++;          // increment the number of cars that have entered the
                                  // street since the last repair
    pthread_cond_signal(&cond);   // signal the condition variable
    pthread_mutex_unlock(&mutex); // unlock the mutex
}

static void outgoing_leave()
{
    pthread_mutex_lock(&mutex);   // lock the mutex
    cars_on_street--;             // decrement the number of cars on the street
    outgoing_onstreet--;          // decrement the number of outgoing cars on the street
    cars_since_repair++;          // increment the number of cars that have entered the
                                  // street since the last repair
    pthread_cond_signal(&cond);   // signal the condition variable
    pthread_mutex_unlock(&mutex); // unlock the mutex
}

/********************END*OF*FUNCTIONS*OF*INTEREST********************/

/* Main code for incoming car threads.
 * You do not need to change anything here, but you can add
 * debug statements to help you during development/debugging.
 */
void *incoming_thread(void *arg)
{
    car *car_info = (car *)arg;

    /* enter street */
    incoming_enter();

    /* Car travel --- do not make changes to the 3 lines below*/
    printf("Incoming car %d has entered and travels for %d minutes\n",
           car_info->car_id, car_info->travel_time);
    travel(car_info->travel_time);
    printf("Incoming car %d has travelled and prepares to leave\n",
           car_info->car_id);

    /* leave street */
    incoming_leave();

    pthread_exit(NULL);
}

/* Main code for outgoing car threads.
 * You do not need to change anything here, but you can add
 * debug statements to help you during development/debugging.
 */
void *outgoing_thread(void *arg)
{
    car *car_info = (car *)arg;

    /* enter street */
    outgoing_enter();

    /* Car travel --- do not make changes to the 3 lines below*/
    printf("Outgoing car %d has entered and travels for %d minutes\n",
           car_info->car_id, car_info->travel_time);
    travel(car_info->travel_time);
    printf("Outgoing car %d has travelled and prepares to leave\n",
           car_info->car_id);

    /* leave street */
    outgoing_leave();

    pthread_exit(NULL);
}

/* Main function sets up simulation and prints report
 * at the end.
 */
int main(int nargs, char **args)
{
    void *status;
    int i, result, num_cars;
    car car_info[MAX_CARS];
    pthread_t street_tid, car_tid[MAX_CARS];

    if (nargs != 2)
    {
        printf("Usage: traffic <name of inputfile>\n");
        return EINVAL;
    }

    num_cars = initialize(car_info, args[1]);
    if (num_cars > MAX_CARS || num_cars <= 0)
    {
        printf(
            "Error:  Bad number of car threads. Maybe there was a problem with "
            "your input file?\n");
        return 1;
    }

    printf("Beginning traffic simulation with %d cars ...\n", num_cars);

    result = pthread_create(&street_tid, NULL, street_thread, NULL);

    if (result)
    {
        printf("traffic:  pthread_create failed for street: %s\n",
               strerror(result));
        exit(1);
    }

    for (i = 0; i < num_cars; i++)
    {
        car_info[i].car_id = i;
        sleep(car_info[i].arrival_time);

        if (strcmp(car_info[i].car_direction, INCOMING) == 0)
            result = pthread_create(&car_tid[i], NULL, incoming_thread,
                                    (void *)&car_info[i]);
        else // car is outgoing
            result = pthread_create(&car_tid[i], NULL, outgoing_thread,
                                    (void *)&car_info[i]);

        if (result)
        {
            printf("traffic: thread_fork failed for car %d: %s\n", i,
                   strerror(result));
            exit(1);
        }
    }

    /* wait for all car threads to finish */
    for (i = 0; i < num_cars; i++)
        pthread_join(car_tid[i], &status);

    /* terminate the street thread. */
    pthread_cancel(street_tid);

    printf("Traffic simulation complete.\n");

    return 0;
}
