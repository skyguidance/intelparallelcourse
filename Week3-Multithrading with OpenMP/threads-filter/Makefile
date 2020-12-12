CXX=icpc
CXXFLAGS=-xMIC-AVX512 -qopenmp -mkl -std=c++11
OPTRPT=-qopt-report=5

default : app


worker.o : worker.cc
	${CXX} -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<"

app : main.cc worker.o
	${CXX} ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" worker.o

clean :
	rm app worker.o *.optrpt

queue:  default
	echo 'cd $$PBS_O_WORKDIR ; ./app' | qsub -l nodes=1:flat -l walltime=00:03:00 -N numintegr
	qstat

clean :
	rm app diffusion.o distribution.o *.optrpt numintegr*