/*
4096
5
100
100
100
100
100
3
*/

/*
Allocator:
( 100 ),( 100 ),
Available mem: 3872
Freelist:
( 100 ),( 100 ),( 100 ),
Available mem (freelist): 300
*/

#include <stdio.h>
#include <stdlib.h>

/**
 * @brief A node in the linked list of free memory blocks.
 *
 */
typedef struct _dll_node
{
    struct _dll_node *pfwd, *pbck;
    int size;
} dll_node;

/**
 * @brief The head of the linked list of free memory blocks.
 *
 */
typedef struct _dll
{
    dll_node *head, *tail;
    int available_memory;
} dll;

void print_list(const dll *);
dll_node *get_new_node(int);
dll_node *remove_from_head(dll *const);
void add_to_tail(dll *, dll_node *, int);

/**
 * @brief main function
 *
 * @return int
 */
int main()
{
    dll list, freelist;
    int available = 0, num_allocs = 0, num_deallocs = 0;

    list.head = NULL;
    list.tail = NULL;
    freelist.head = NULL;
    freelist.tail = NULL;
    freelist.available_memory = 0; // freelist is empty

    // first line is available memory given in bytes
    scanf("%d", &available);
    list.available_memory = available - 8;

    // second line is the number of allocations
    scanf("%d", &num_allocs);

    for (int i = 0; i < num_allocs; ++i)
    {
        int alloc = 0;
        scanf("%d", &alloc);
        add_to_tail(&list, get_new_node(alloc), 0);
    }

    // following line is the number of deallocations
    scanf("%d", &num_deallocs);
    for (int i = 0; i < num_deallocs; ++i)
        add_to_tail(&freelist, remove_from_head(&list), 1);

    puts("Allocator:");
    print_list(&list);
    printf("\nAvailable mem: %d\n", list.available_memory);

    puts("Freelist:");
    print_list(&freelist);
    printf("\nAvailable mem (freelist): %d\n", freelist.available_memory);
    return 0;
}

/**
 * @brief Get the new node object
 *
 * @param list
 * @param new
 * @param is_freelist
 */
void add_to_tail(dll *list, dll_node *new, int is_freelist)
{
    // add code here
    if (new == NULL)
    {
        fprintf(stderr, "Error: add_to_tail: new is NULL\n");
        exit(1);
    }

    if (list == NULL)
    {
        fprintf(stderr, "Error: add_to_tail: list is NULL\n");
        exit(1);
    }

    new->pfwd = NULL;

    if (list->tail == NULL)
    {
        list->head = new;
        list->tail = new;
        new->pbck = NULL;
    }
    else
    {
        list->tail->pfwd = new;
        new->pbck = list->tail;
        list->tail = new;
    }
    list->available_memory = list->available_memory +
                             (2 * is_freelist - 1) * new->size +
                             (is_freelist - 1) * 8;
}

/**
 * @brief Remove the node from the list
 *
 * @param list
 * @return dll_node*
 */
dll_node *remove_from_head(dll *const list)
{
    // add code here
    if (list == NULL)
    {
        fprintf(stderr, "Error: remove_from_head: list is NULL\n");
        exit(1);
    }
    if (list->head == NULL)
        return NULL;

    dll_node *temp = list->head;

    if (list->head == list->tail)
    {
        list->head = NULL;
        list->tail = NULL;
    }
    else
    {
        list->head = list->head->pfwd;
        list->head->pbck = NULL;
        list->available_memory += temp->size + 8;
    }
    return temp;
}

/**
 * @brief Print the list
 *
 * @param list
 */
void print_list(const dll *const list)
{
    dll_node *temp = list->head;
    if (temp == NULL)
        fprintf(stdout, " ");
    else
        while (temp != NULL)
        {
            printf("( %d ),", temp->size);
            temp = temp->pfwd;
        }
}

/**
 * @brief Get the new node object
 *
 * @param size
 * @return dll_node*
 */
dll_node *get_new_node(int size)
{
    dll_node *new = (dll_node *)malloc(size + 8);
    if (new == NULL)
    {
        fprintf(stderr, "Error: get_new_node: malloc failed");
        exit(1);
    }
    new->size = size;
    new->pfwd = NULL;
    new->pbck = NULL;
    return new;
}