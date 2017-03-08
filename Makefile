all: build
	make -C src 
clean:
	make -C src clean 
test:
	echo "Test"
build:
	mkdir build

