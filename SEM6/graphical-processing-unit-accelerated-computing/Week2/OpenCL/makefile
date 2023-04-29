CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lOpenCL

SOURCE = AddTwoNumberOnGPU.c
SOURCE = InitDataOnGPU.c
SOURCE = Calculator.c

TARGET = executable

all: $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)