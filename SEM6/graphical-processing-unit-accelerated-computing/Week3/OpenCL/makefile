CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL 

SOURCE = CalculateSumOnGPU.c
SOURCE = CalculateSumOnGPU_BlockIdx.c
SOURCE = Calculate_SumOnGPU_Timed.c

$(SOURCE:.c=): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	./$@

clean:
	rm -f $(SOURCE:.c=) *.o *.out *.exe 
