CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: main

main: main.cpp
	g++ $(CFLAGS) main.cpp -o main $(LIBS)

clean:
	rm main