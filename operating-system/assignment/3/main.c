#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "parse.c"

#define MAX 100

void CMD(parseInfo *info, char *myargs[], char *history[])
{
    int index = atoi(myargs[1]);

    if (index < 0 || index >= MAX)
        printf("Invalid index\n");
    else if (history[index] == NULL)
        printf("No command at index %d\n", index);
    else
    {
        if (strcmp(history[index], "!CMD") == 0)
        {
            printf("Cannot execute !CMD inside !CMD\n");
            exit(0);
        }
        else
        {
            info = parse(history[index]);
            for (int i = 0; i < 11; i++)
                myargs[i] = info->CommArray[0].VarList[i];
        }
    }
}

void help()
{
    printf(
        "1. jobs - provides a list of all background processes and their local "
        "pIDs.\n");
    printf("2. cd PATHNAME - sets the PATHNAME as working directory.\n");
    printf(
        "3. history - prints a list of previously executed commands. Assume 10 "
        "as the maximum history\n");
    printf(
        "4. kill PID - terminates the background process identified locally "
        "with PID in the jobs list.\n");
    printf("5. !CMD - runs the command numbered CMD in the command history.\n");
    printf(
        "6. exit - terminates the shell only if there are no background "
        "jobs.\n");
    printf(
        "7. help - prints the list of builtin commands along with their "
        "description.\n");
}

char *cd(char *myargs[])
{
    if (myargs[1] == NULL)
    {
        if (chdir(getenv("HOME")) == -1)
            perror("chdir() to $HOME failed\n");
    }
    else if (chdir(myargs[1]) == -1)
        perror("chdir() failed\n");
    return getcwd(NULL, MAX);
}

void history(char *history[])
{
    for (int i = 0; i < MAX; i++)
        if (history[i] != NULL)
            printf("%d : %s", i, history[i]);
}

void jobs()
{
    char *args[] = {"ps", "-A", "-o", "pid,ppid,stat,command", NULL};
    if (execvp(args[0], args) == -1)
        perror("execvp() failed\n");
}

int main(int argc, char *argv[])
{
    int infile, outfile;
    char *dir = getcwd(NULL, MAX), *H[MAX];
    unsigned int counter = 0;

    while (1)
    {
        printf("%s> ", dir);

        char *cmd = (char *)malloc(MAX * sizeof(char));
        fgets(cmd, MAX, stdin);

        H[counter % MAX] = cmd;
        counter++;

        parseInfo *info = parse(cmd);

        char *myargs[11];
        for (int i = 0; i < 11; i++)
            myargs[i] = info->CommArray[0].VarList[i];

        if (strcmp(myargs[0], "exit") == 0)
            return 0;

        int pid = fork();

        if (pid == -1)
            fprintf(stderr, "An errored occured while forking the process %d\n",
                    pid);
        else if (pid == 0)
        {
            if (strcmp(myargs[0], "!CMD") == 0)
                CMD(info, myargs, H);

            if (info->boolInfile)
            {
                if (access(info->inFile, F_OK) == 0)
                {
                    if (access(info->inFile, R_OK) == 0)
                    {
                        infile = open(info->inFile, O_RDONLY);
                        dup2(infile, fileno(stdin));
                        if (close(infile) == -1)
                            fprintf(stderr, "Error closing file\n");
                    }
                    else
                        continue;
                }
                else
                    printf("File does not exist\n");
            }

            if (info->boolOutfile)
            {
                if (access(info->inFile, F_OK) == 0)
                {
                    if (access(info->inFile, W_OK) == 0)
                    {
                        outfile = open(info->outFile, O_WRONLY);
                        dup2(outfile, fileno(stdout));
                        if (close(outfile) == -1)
                            fprintf(stderr, "Error closing file\n");
                    }
                    else
                        continue;
                }
                else
                    printf("File does not exist\n");
            }

            if (strcmp(myargs[0], "help") == 0)
                help();

            else if (strcmp(myargs[0], "cd") == 0)
                dir = cd(myargs);

            else if (strcmp(myargs[0], "history") == 0)
                history(H);

            else if (strcmp(myargs[0], "jobs") == 0)
                jobs();

            else if (execvp(myargs[0], myargs) == -1)
            {
                perror("execvp");
                exit(1);
            }
        }
        else if (info->boolBackground == 0)
            wait(NULL);
    }

    return 0;
}
