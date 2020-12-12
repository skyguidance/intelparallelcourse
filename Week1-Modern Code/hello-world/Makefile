TARGET=main.cc

default: app-icc app-gcc

app-icc : ${TARGET}
	# Insert Intel compiler compilation here

app-gcc : ${TARGET}
	g++ -o "$@" "$<"

clean :
	rm app-icc app-gcc