EXECS = piconn 
BUILD = ../build/

all: $(EXECS)

piconn: piconn.cpp
	g++ -DNDEBUG -Wall $< -o ${BUILD}$@ -O3 -std=c++0x -g

piconn_debug: piconn.cpp
	g++ -Wall $< -o ${BUILD}$@ -O0 -std=c++0x -g

clean:
	rm -f ${BUILD}$(EXECS) ${BUILD}piconn_debug
