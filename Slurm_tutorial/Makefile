CC = nvcc
CFLAGS = -O3 -arch=sm_89
TARGET = gpu_test

all: $(TARGET)

$(TARGET): gpu_test.cu
	$(CC) $(CFLAGS) -o $(TARGET) gpu_test.cu

clean:
	rm -f $(TARGET) *.o
