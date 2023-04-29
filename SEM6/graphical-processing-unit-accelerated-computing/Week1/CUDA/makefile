CC = nvcc
CFLAGS = -Wall -Wextra 
SOURCE = HelloWorldGPU.cu
TARGET = HelloWorldGPU

all: $(TARGET)
	$(CC) $(CFLAGS) $(SOURCE) -o $(TARGET)	

run: $(TARGET)
	./$(TARGET)

clean: 
	rm -f $(TARGET)