FC=gfortran
FFLAGS="-g -fcheck=all"
LDFLAGS=-llapack -lblas
#OPTFLAGS=-O2
OPTFLAGS=-O0
ARCHFLAGS=-march=native

.PHONY: clean

all: lib_linear_regression.so lib_ridge_regression.so lib_gaussian_naive_bayes.so

lib_%.so: %.f90
	f2py $(LDFLAGS) -c $^ -m $(basename $@) --f90flags=$(FFLAGS) --opt="$(OPTFLAGS)" --arch="$(ARCHFLAGS)"

clean:
	rm -f *.so