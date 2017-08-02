objects = multNoShare.o 

all: $(objects)
	nvcc -g -arch=compute_30 $(objects) -o app

%.o: %.c
	nvcc -x cu -arch=compute_30 -I. -dc $< -o $@
