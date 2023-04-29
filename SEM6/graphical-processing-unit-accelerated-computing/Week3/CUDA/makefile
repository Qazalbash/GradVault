CC = nvcc
SOURCE = CalculateSumOnGPU.c

$(SOURCE:.c=): $(SOURCE)
	$(CC) -o $@ $^
	./$@

clean:
	rm -f $(SOURCE:.c=)