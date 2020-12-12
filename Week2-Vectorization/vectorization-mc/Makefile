CXX=icpc
CXXFLAGS=-xMIC-AVX512 -qopenmp-simd -mkl
OPTRPT=-qopt-report=5

default : app

distribution.o : distribution.cc distribution.h
	icpc -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<"

diffusion.o : diffusion.cc distribution.o
	icpc -c ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" distribution.o

app : main.cc diffusion.o distribution.o
	icpc ${OPTRPT} ${CXXFLAGS} -o "$@" "$<" diffusion.o distribution.o

queue:  default
	echo 'cd $$PBS_O_WORKDIR ; ./app' | qsub -l nodes=1:flat -l walltime=00:03:00 -N numintegr
	qstat

clean :
	rm app diffusion.o distribution.o *.optrpt numintegr*
