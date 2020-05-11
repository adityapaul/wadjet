CC=g++
CFLAGS=-std=c++11 -g -I/usr/local/opt/openblas/include `pkg-config opencv4 --cflags --libs`

train: train.cpp Feature.cpp
	$(CC) $(CFLAGS) -o train train.cpp Feature.cpp 

test: test.cpp Feature.cpp
	$(CC) $(CFLAGS) -o test test.cpp Feature.cpp

clean:
	rm train test
