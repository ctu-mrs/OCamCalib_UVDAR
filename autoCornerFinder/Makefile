
INCS = 
LIBS = 


DEFINCS = -I/usr/local/include -I/usr/include/opencv4
DEFLIBS = -L/usr/local/lib/ -L/usr/lib/x86_64-linux-gnu/
LINKLIBS = -lopencv_core -lopencv_highgui -lopencv_imgproc  -lopencv_imgcodecs -lopencv_videoio

CPP  = g++ -std=c++11
CC   = gcc
OBJ  = main.o cvcalibinit3.o 
LINKOBJ  = main.o cvcalibinit3.o 
BIN  = FindCorners
RM = rm -f


all: $(BIN)

clean:
	${RM} $(OBJ) $(BIN)

$(BIN): $(OBJ)
	$(CPP) $(LINKOBJ) -g -o $(BIN)  $(LIBS) $(DEFLIBS) $(LINKLIBS)

main.o: main.cpp
	$(CPP) -c -g main.cpp -o main.o $(INCS) $(DEFINCS) 

cvcalibinit3.o: cvcalibinit3.cpp
	$(CPP) -c -g cvcalibinit3.cpp -o cvcalibinit3.o  $(INCS) $(DEFINCS)
