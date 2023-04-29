CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL
SOURCE = MatrixMultiplication_Task.c
SOURCE = Simple2D.c
SOURCE = RawImageProcessing.c
TARGET = exec

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)