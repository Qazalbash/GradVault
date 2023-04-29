CC = nvcc
CFLAGS = -Wall -Wextra -Werror -O3
SOURCE = q1.cu 
TARGET = main

all: $(TARGET)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)