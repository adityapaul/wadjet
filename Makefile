CC=g++
CFLAGS=-O3 -std=c++11 -I/usr/local/opt/openblas/include `pkg-config opencv4 --cflags --libs`

train: train.cpp Feature.cpp
	$(CC) $(CFLAGS) -o train train.cpp Feature.cpp 

clean:
	rm train
