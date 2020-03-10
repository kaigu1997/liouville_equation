CXX := icpc
WARNINGFLAGS := -Wall -Wextra -Wconversion -Wshadow
CXXFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -O3 -g
LDFLAGS := ${WARNINGFLAGS} -mkl -std=c++17 -O3 -Wl,-fuse-ld=gold
Objects := matrix.o general.o pes.o main.o
HeaderFile := matrix.h general.h pes.h

.PHONY: all
all: mqcl

mqcl: ${Objects}
	${CXX} ${LDFLAGS} ${Objects} -o mqcl
main.o: main.cpp ${HeaderFile}
	${CXX} ${CXXFLAGS} -c main.cpp -o main.o
pes.o: pes.cpp ${HeaderFile}
	${CXX} ${CXXFLAGS} -c pes.cpp -o pes.o
general.o: general.cpp ${HeaderFile}
	${CXX} ${CXXFLAGS} -c general.cpp -o general.o
matrix.o: matrix.cpp matrix.h
	${CXX} ${CXXFLAGS} -c matrix.cpp -o matrix.o

.PHONY: clean
clean:
	-rm *.o

.PHONY: clean_result
clean_result:
	-rm log output *.txt *.png *.gif

.PHONY: git
git:
	git add *.h *.cpp makefile *.sh *.py .gitignore

