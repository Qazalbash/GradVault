CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL
SOURCE_MATRIX = MatrixMultiplicationTiled.c
SOURCE_SMBM = SharedMemoryBitmap.c
TARGET_MATRIX = MatrixMultiplicationTiled
TARGET_SMBM = SharedMemoryBitmap

$(TARGET_MATRIX): $(SOURCE_MATRIX)
	$(CC) $(CFLAGS) $(SOURCE_MATRIX) -o $(TARGET_MATRIX) $(LDFLAGS)

$(TARGET_SMBM): $(SOURCE_SMBM)
	$(CC) $(CFLAGS) $(SOURCE_SMBM) -o $(TARGET_SMBM) $(LDFLAGS)

all: $(TARGET_MATRIX) $(TARGET_SMBM)

clean:
	rm -f $(TARGET_MATRIX) $(TARGET_SMBM)