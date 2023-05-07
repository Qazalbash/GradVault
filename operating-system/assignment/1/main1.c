#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    char processID[10], priority[7];
    int arrivalTime, duration, tickets, ticketsArray[3];
} container;

int main()
{
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    srand(1);
    char inputLine[20], selectedProcess[3];
    int noOfProcess, i, j, k, m, n, totalTime = 1, winTicket, flag = 0,
                                    counter = 0, total_tickets = 0;
    container process, tmpProcess, allProcesses[255];

    fscanf(stdin, "%d", &noOfProcess);

    for (i = 0; i < noOfProcess; ++i)
    {
        fscanf(stdin, "%s", inputLine);

        strcpy(process.processID, strtok(inputLine, ":"));
        process.arrivalTime = atoi(strtok(NULL, ":"));
        process.duration = atoi(strtok(NULL, ":"));
        strcpy(process.priority, strtok(NULL, ":"));

        totalTime = totalTime + process.duration;

        allProcesses[i] = process;
    }

    // SORTING TICKETS FOR THE FIRST TIME ONTHE BASIS OF ARRIVAL TIME
    for (i = 0; i < noOfProcess; ++i)
    {
        for (j = i + 1; j < noOfProcess; ++j)
        {
            if (allProcesses[i].arrivalTime > allProcesses[j].arrivalTime)
            {
                tmpProcess = allProcesses[i];
                allProcesses[i] = allProcesses[j];
                allProcesses[j] = tmpProcess;
            }
        }
    }

    for (i = 1; i < totalTime; ++i)
    {
        counter = 0;
        total_tickets = 0;

        printf("%d:", i);
        for (int a = 0; a < noOfProcess; ++a)
        {
            if (allProcesses[a].arrivalTime <= i)
            {
                flag = 0;
                if (strcmp(allProcesses[a].priority, "low") == 0)
                {
                    flag = 1;
                    allProcesses[a].tickets = 1;
                }
                else if (strcmp(allProcesses[a].priority, "normal") == 0)
                {
                    flag = 1;
                    allProcesses[a].tickets = 2;
                }
                else if (strcmp(allProcesses[a].priority, "high") == 0)
                {
                    flag = 1;
                    allProcesses[a].tickets = 3;
                }

                if (flag)
                    for (k = 0; k < allProcesses[a].tickets; ++k)
                        allProcesses[a].ticketsArray[k] = total_tickets++;
                // total_tickets += 1;
            }
        }

        if (total_tickets > 0)
        {
            winTicket = rand() % total_tickets;
            for (m = 0; m < noOfProcess; m++)
                for (n = 0; n < allProcesses[m].tickets; n++)
                    if (winTicket == allProcesses[m].ticketsArray[n])
                    {
                        strcpy(selectedProcess, allProcesses[m].processID);
                        allProcesses[m].duration--;
                        printf("%s:", selectedProcess);
                        counter++;
                    }

            flag = 0;

            for (m = 0; m < noOfProcess; m++)
            {
                if (allProcesses[m].duration == 0)
                {
                    strcpy(allProcesses[m].priority, "done");
                    strcpy(allProcesses[m].processID, "done");
                    allProcesses[m].tickets = 0;
                }
                else if (allProcesses[m].duration >= 0 &&
                         strcmp(allProcesses[m].processID, selectedProcess) !=
                             0 &&
                         allProcesses[m].arrivalTime <= i &&
                         strcmp(allProcesses[m].processID, "done") != 0)
                {
                    printf("%s,", allProcesses[m].processID);
                    flag = 1;
                    counter++;
                }
            }
            if (flag == 0)
                printf("empty");
            printf("\n");
        }
        if (total_tickets == 0)
        {
            printf("idle:empty\n");
            totalTime++;
        }
    }
}