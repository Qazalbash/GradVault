/*
3
P1:2:6:low
P2:5:3:normal
P3:2:7:high
*/

/*
1:idle:empty
2:P3:P1,
3:P3:P1,
4:P3:P1,
5:P3:P1,P2,
6:P2:P1,P3,
7:P3:P1,P2,
8:P2:P1,P3,
9:P1:P3,P2,
10:P3:P1,P2,
11:P3:P1,P2,
12:P2:P1,
13:P1:empty
14:P1:empty
15:P1:empty
16:P1:empty
17:P1:empty
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct process
{
    char *process_id;
    char *priority;
    unsigned int duration;
    unsigned int arrival_time;
    unsigned int tickets;
    struct process *next;
} process;

void insert_process(process **, char *, char *, const unsigned int,
                    const unsigned int, const unsigned int);
void remove_process(process **, const char *);
void print_waiting_Q(const process *, const char *);
void update_tickets(process *, char *[]);
void update_duration(process *, const void *, unsigned int *);

int main(int argc, char *argv[])
{
    srand(1); // seeding the random number

    unsigned int total_process = 0; // total processes
    fscanf(stdin, "%d\n", &total_process);

    // Initializing the waiting queue
    process *front = NULL;

    char *current_process;                          // running process
    char *tickets_container[3 * total_process + 1]; // array for tickets
    char line[total_process][255];                  // line to get input
    char *priorities[total_process];                // priorities
    char *process_ids[total_process];               // process ids

    unsigned int tickets;                     // tickets for one process
    unsigned int time = 1;                    // time
    unsigned int p_counter = 0;               // process counter
    unsigned int total_tickets = 0;           // total tickets
    unsigned int duration[total_process];     // process duration
    unsigned int arrival_time[total_process]; // arrival time
    unsigned int *total_tickets_ptr =
        &total_tickets; // pointer to store all the tickets value

    for (unsigned int i = 0; i < total_process; i++) // taking input as process
    {
        fgets(line[i], sizeof(line[i]), stdin);

        *(process_ids + i) = strtok(line[i], ":");     // processID
        *(arrival_time + i) = atoi(strtok(NULL, ":")); // arrival time
        *(duration + i) = atoi(strtok(NULL, ":"));     // duration
        *(priorities + i) = strtok(NULL, "\n");        // priority
    }

    while (p_counter < total_process || front != NULL)
    {
        for (unsigned int j = 0; j < total_process; j++)
        {
            if (*(arrival_time + j) == time)
            {
                /*
                this is the branchless way to do arithematics
                if you want to see it you can check out my article
                https://mesumali26-ma.medium.com/branchless-programming-with-python-124a85bd7481
                */

                tickets = 3 * (strncmp(*(priorities + j), "high", 4) == 0) +
                          2 * (strncmp(*(priorities + j), "normal", 6) == 0) +
                          (strncmp(*(priorities + j), "low", 3) == 0);

                insert_process(&front, *(process_ids + j), *(priorities + j),
                               *(duration + j), *(arrival_time + j), tickets);

                total_tickets += tickets;

                p_counter++;
            }
        }

        update_tickets(front, tickets_container);

        current_process = (p_counter > 0)
                              ? (tickets_container[rand() % total_tickets])
                              : "idle";

        printf("%d:%s:", time, current_process);

        print_waiting_Q(front, current_process);

        update_duration(front, current_process, total_tickets_ptr);

        if (p_counter == total_process && total_tickets == 0)
            break;

        time++;
    }

    return 0;
}

void insert_process(process **front, char *process_id, char *priority,
                    const unsigned int duration,
                    const unsigned int arrival_time,
                    const unsigned int tickets)
{
    process *n = malloc(sizeof(process)); // creating heap memory for new node

    // assigning the values
    n->process_id = process_id;
    n->priority = priority;
    n->duration = duration;
    n->arrival_time = arrival_time;
    n->tickets = tickets;
    n->next = NULL;

    if (*front == NULL) // list is empty
        *front = n;
    else
    {
        process *tmp = *front;
        while (tmp->next != NULL)
            tmp = tmp->next;
        n->next = tmp->next;
        tmp->next = n;
    }
}

void remove_process(process **front, const char *process_id)
{
    process *tmp = *front;

    if (tmp->next == NULL && strcmp(tmp->process_id, process_id) == 0)
        free(tmp);
    else if (tmp->process_id == process_id)
        **front = *((*front)->next);
    else
        while (tmp->next != NULL)
        {
            if (strcmp(tmp->next->process_id, process_id) == 0)
            {
                process *tmp2 = tmp->next;
                tmp->next = tmp->next->next;
                free(tmp2);
            }
            else
                tmp = tmp->next;
        }
}

void print_waiting_Q(const process *front, const char *current_process)
{
    if (front == NULL)
        fprintf(stdout, "empty\n");
    else if (front->next == NULL)
        fprintf(stdout, "empty\n");
    else
    {
        while (front != NULL)
        {
            if (front->process_id != current_process)
                printf("%s,", front->process_id);
            front = front->next;
        }
        printf("\n");
    }
}

void update_tickets(process *front, char *ticket_container[])
{
    unsigned int i = 0;
    while (front != NULL)
    {
        for (int j = i; j < (i + (front->tickets)); j++)
            *(ticket_container + j) = (front->process_id);
        i += front->tickets;
        front = front->next;
    }
}

void update_duration(process *front, const void *desired_process,
                     unsigned int *total_tickets)
{
    process *tmp = front;
    if (strncmp(desired_process, "idle", 4))
        while (front != NULL)
        {
            if (desired_process == front->process_id)
            {
                front->duration--;
                if (front->duration == 0)
                {
                    *total_tickets -= front->tickets;
                    remove_process(&tmp, front->process_id);
                    break;
                }
            }
            front = front->next;
        }
}