CXX := mpicxx
CFLAGS := -Wall -O3
all: nbody.out
nbody.out: nbody.cpp
	$(CXX) $(CFLAGS) -o $@ $^
