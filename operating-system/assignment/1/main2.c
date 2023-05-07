#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct
{
    char processID[10];
    int arrivalTime;
    int duration;
    char priority[10];
    int e;
    int f[3];
    int endTime;
    int priorityflag;
    int isFinish;
} tuple;

int main()
{
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */
    int noOfentries, i, j, k, TotalTime, TotalTickets, aTime, count, winTicket,
        temp, temp1;
    int flag = 0;
    char inpt[30];
    char priority[100];
    tuple L[1000];
    tuple t, a;
    TotalTime = 1;
    TotalTickets = 0;
    srand(1);

    fscanf(stdin, "%d", &noOfentries);
    // fprintf(stdout,"%d\n", noOfentries);
    for (i = 0; i < noOfentries; ++i)
    {
        fscanf(stdin, "%s", inpt);
        char *token = strtok(inpt, ":");
        for (j = 0; j < 4; ++j)
        {
            if (j == 0)
            {
                strcpy(t.a, token);
                // printf("%s\n",t.a);
                token = strtok(NULL, ":");
            }
            else if (j == 1)
            {
                t.b = atoi(token);
                // printf("%d\n", t.b);
                token = strtok(NULL, ":");
            }
            else if (j == 2)
            {
                t.c = atoi(token);
                TotalTime = TotalTime + t.c;
                // printf("%d\n", t.c);
                token = strtok(NULL, ":");
            }
            else
            {
                strcpy(t.d, token);
                // printf("%s\n", t.d);
                token = strtok(NULL, ":");
                t.endTime = t.b + t.c;
            }
        }
        t.priorityflag = 0;
        t.isFinish = 0;
        L[i] = t;
    }

    // SORTING TICKETS FOR THE FIRST TIME ONTHE BASIS OF ARRIVAL TIME
    for (i = 0; i < noOfentries; ++i)
    {
        for (j = i + 1; j < noOfentries; ++j)
        {
            if (L[i].b > L[j].b)
            {
                a = L[i];
                L[i] = L[j];
                L[j] = a;
            }
        }
    }
    for (int i = 1; i < TotalTime; ++i)
    {
        int counter = 0;
        int Total_tickets = 0;
        // printf("Timer");
        printf("%d:", i);
        int wicket = 0;
        for (int a = 0; a < noOfentries; ++a)
        {
            if (L[a].b <= i)
            {
                if (strcmp(L[a].d, "low") == 0)
                {
                    L[a].e = 1;
                    for (int j = 0; j < 1; j++)
                    {
                        L[a].f[j] = Total_tickets;
                        Total_tickets += 1;
                    }
                }
                else if (strcmp(L[a].d, "normal") == 0)
                {
                    L[a].e = 2;
                    for (int j = 0; j < 2; ++j)
                    {
                        L[a].f[j] = Total_tickets;
                        Total_tickets += 1;
                    }
                }
                else if (strcmp(L[a].d, "high") == 0)
                {
                    L[a].e = 3;
                    for (int j = 0; j < 3; ++j)
                    {
                        L[a].f[j] = Total_tickets;
                        Total_tickets += 1;
                    }
                }
            }
        }

        // choosing winticket
        if (Total_tickets > 0)
        {
            winTicket = rand() % Total_tickets;
            char chosen_process[3];
            // printf(" win ticket: %d ", winTicket);
            for (int m = 0; m < noOfentries; m++)
            {
                for (int n = 0; n < L[m].e; n++)
                {
                    // printf(" idk %d: ", L[m].f[n]);
                    if (winTicket == L[m].f[n])
                    {
                        strcpy(chosen_process, L[m].a);
                        L[m].c--;
                        // printf("Chosen process");
                        printf("%s:", chosen_process);
                        counter++;
                    }
                    // printf("FAMA");
                }
            }
            flag = 0;
            for (int o = 0; o < noOfentries; o++)
            {
                if (L[o].c == 0)
                {
                    strcpy(L[o].a, "xx");
                    strcpy(L[o].d, "xx");
                    L[o].e = 0;
                }
                else if (L[o].c >= 0 && strcmp(L[o].a, chosen_process) != 0 &&
                         L[o].b <= i && strcmp(L[o].a, "xx") != 0)
                {
                    printf("%s,", L[o].a);
                    flag = 1;
                    counter++;
                }
            }
            if (flag == 0)
            {
                printf("empty");
            }
            printf("\n");
        }
        if (Total_tickets == 0)
        {
            printf("idle:empty \n");
            TotalTime++;
        }
    }
}