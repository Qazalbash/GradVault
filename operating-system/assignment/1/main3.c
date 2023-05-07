/*
3
P1:2:6:low
P2:5:3:normal
P3:2:7:high
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct process
{
    char *process_id, *priority;
    unsigned int duration, arrival_time, tickets;
    struct process *next;
} process;

void insert_process(process **, char *, char *, unsigned int, unsigned int,
                    unsigned int);
void remove_process(process **, char *);
void print_waiting_Q(process *, char *);
void update_tickets(process *, void *[], int);
void update_duration(process *, void *, unsigned int *);

int main(int argc, char *argv[])
{
    srand(1); // seeding the random number

    unsigned int *total_process = malloc(sizeof(int)); // total processes
    fscanf(stdin, "%d\n", total_process);

    // Initializing the wainting queue
    process *front = NULL;

    void *tickets_container[100]; // array for tickets

    char *current_process;             // running process
    char line[*total_process][255];    // line to get input
    char *priorities[*total_process];  // priorities
    char *process_ids[*total_process]; // process ids

    unsigned int tickets;                      // tickets for one process
    unsigned int time = 1;                     // time
    unsigned int p_counter = 0;                // process counter
    unsigned int total_tickets = 0;            // total tickets
    unsigned int arrival_time[*total_process]; // arrival time
    unsigned int duration[*total_process];     // process duration

    for (unsigned int i = 0; i < *total_process;
         i++) // taking input as process
    {
        fgets(line[i], sizeof(line[i]), stdin);

        *(process_ids + i) = strtok(line[i], ":");     // processID
        *(arrival_time + i) = atoi(strtok(NULL, ":")); // arrival time
        *(duration + i) = atoi(strtok(NULL, ":"));     // duration
        *(priorities + i) = strtok(NULL, "\n");        // priority
    }

    while (p_counter < *total_process || front != NULL)
    {
        for (unsigned int j = 0; j < *total_process; j++)
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

        update_tickets(front, tickets_container, total_tickets);

        current_process = (p_counter > 0)
                              ? (tickets_container[rand() % total_tickets])
                              : "idle";

        printf("%d:%s:", time, current_process);
        print_waiting_Q(front, current_process);

        unsigned int *totalTicketsPtr = &total_tickets;

        update_duration(front, current_process, totalTicketsPtr);

        time++;

        if (p_counter == *total_process && total_tickets == 0)
        {
            free(total_process);
            return 0;
        }
    }

    free(total_process);
    return 0;
}

void insert_process(process **headaddr, char *processId, char *priority,
                    unsigned int duration, unsigned int arrival_time,
                    unsigned int tickets)
{
    process *n = malloc(sizeof(process)); // creating heap memory for new node

    // assigning the values
    n->process_id = processId;
    n->priority = priority;
    n->duration = duration;
    n->arrival_time = arrival_time;
    n->tickets = tickets;
    n->next = NULL;

    if (*headaddr == NULL) // list is empty
        *headaddr = n;
    else
    {
        process *tmp = *headaddr;
        while (tmp->next != NULL)
            tmp = tmp->next;
        n->next = tmp->next;
        tmp->next = n;
    }
}

void remove_process(process **headaddr, char *desiredProcess)
{
    process *tmp = *headaddr;

    if (tmp->next == NULL && strcmp(tmp->process_id, desiredProcess) == 0)
        free(tmp);
    else if (tmp->process_id == desiredProcess)
        **headaddr = *((*headaddr)->next);
    else
        while (tmp->next != NULL)
        {
            if (strcmp(tmp->next->process_id, desiredProcess) == 0)
            {
                process *tmp2 = tmp->next;
                tmp->next = tmp->next->next;
                free(tmp2);
                return;
            }
            tmp = tmp->next;
        }
}

void print_waiting_Q(process *headaddr, char *runningProcess)
{
    if (headaddr == NULL)
        fprintf(stdout, "empty\n");
    else if (headaddr->next == NULL)
        fprintf(stdout, "empty\n");
    else
    {
        while (headaddr != NULL)
        {
            if (headaddr->process_id != runningProcess)
                printf("%s,", headaddr->process_id);
            headaddr = headaddr->next;
        }
        printf("\n");
    }
}

void update_tickets(process *head, void *ticketsContainer[], int totalTickets)
{
    unsigned int index = 0;
    while (head != NULL)
    {
        for (int w = index; w < (index + (head->tickets)); w++)
            *(ticketsContainer + w) = (head->process_id);
        index += head->tickets;
        head = head->next;
    }
}

void update_duration(process *head, void *desiredProcess,
                     unsigned int *totalTickets)
{
    process *actualHead = head;
    if (strncmp(desiredProcess, "idle", 4) == 0)
    {
        return;
    }
    while (head != NULL)
    {
        if (desiredProcess == head->process_id)
        {
            head->duration--;
            if (head->duration == 0)
            {
                *totalTickets -= head->tickets;
                remove_process(&actualHead, head->process_id);
                return;
            }
        }
        head = head->next;
    }
}