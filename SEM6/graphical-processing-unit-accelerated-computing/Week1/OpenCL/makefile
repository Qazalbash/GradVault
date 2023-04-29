CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -lOpenCL

SOURCE = Hello1.c 
SOURCE = Hello2.c
SOURCE = Hello3.c 
SOURCE = main.c 
SOURCE = OpenCLtemplate.c 
SOURCE = hello.c

TARGET = executable

all: $(SOURCE)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)