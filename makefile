CC=g++
CFLAGS=-I.
OS := $(shell uname)

dsverifier: src/dsverifier.cpp

ifeq ($(shell uname),Darwin)
	$(CC) -o dsverifier -std=c++11 src/dsverifier.cpp -I /usr/local/include/eigen3/ -I. -I /opt/local/include/
else
	$(CC) -o dsverifier -static -std=c++11 src/dsverifier.cpp -I /usr/include/eigen3/ -I.
endif

clean:
	rm -rf dsverifier

bmc-download:
	@echo "Downloading CBMC 5.7"
	@lwp-download http://www.cprover.org/cbmc/download/cbmc-5-7-linux-64.tgz
	@mkdir cbmc-5-7-linux-64
	@tar -xvzf cbmc-5-7-linux-64.tgz -C cbmc-5-7-linux-64
	@rm -Rf model-checker/
	@mkdir model-checker
	@mv cbmc-5-7-linux-64/cbmc model-checker/
	@rm cbmc-5-7-linux-64.tgz
	@rm -Rf cbmc-5-7-linux-64/
	@echo "Downloading ESBMC 4.4.1"
	@lwp-download http://ssvlab.hussama.io/binaries/esbmc-v4.4.1-linux-static-64.tgz
	@tar -xvzf esbmc-v4.4.1-linux-static-64.tgz
	@mv esbmc-v4.4.1-linux-static-64/bin/esbmc model-checker/
	@rm esbmc-v4.4.1-linux-static-64.tgz
	@rm -Rf esbmc-v4.4.1-linux-static-64/
