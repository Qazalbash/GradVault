CC = gcc
CFLAGS = -Wall -Wextra -g
LDFLAGS = -lm
SOURCE = q1a.c
TARGET = q1a

INPUT_IMAGE = lena.img
OUTPUT_IMAGE = lena_out.img
WIDTH = 256
HEIGHT = 256
METHOD = 2
ROTATION = 15

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	./$(TARGET) $(INPUT_IMAGE) $(OUTPUT_IMAGE) $(WIDTH) $(HEIGHT) $(METHOD) $(ROTATION)
	python3 viewer.py
	code lena_out.png

run: $(TARGET) $(TARGET).c
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	./$(TARGET) $(INPUT_IMAGE) $(OUTPUT_IMAGE) $(WIDTH) $(HEIGHT) $(METHOD) $(ROTATION)
	python3 viewer.py
	code lena_out.png

clean:
	rm -f $(TARGET)
