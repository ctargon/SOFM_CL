# The makefile for MP2.
# Type:
#   make         -- to build program lab2
#   make testlist -- to compile testing program
#   make clean   -- to delete object files, executable, and core
#   make design  -- check for simple design errors (incomplete)
#   make list.o  -- to compile only list.o (or: use lab2.o, hpot_support.o)
#
# You should not need to change this file.  
#
# Format for each entry
#    target : dependency list of targets or files
#    <tab> command 1
#    <tab> command 2
#    ...
#    <tab> last command
#    <blank line>   -- the list of commands must end with a blank line

sofm_cl : main_cl.o sofm.o AOCL_Utils.o 
	g++ -Wall -g main_cl.o sofm.o AOCL_Utils.o -o sofm_cl -lm -lOpenCL #-framework OpenCL

sofm_serial : main_serial.o sofm.o
	g++ -Wall -g main_serial.o sofm.o -o sofm_serial -lm

main_cl.o : ./host/src/main.cpp ./host/src/sofm.h ./common/inc/AOCL_Utils.h 
	g++ -Wall -g -c ./host/src/main.cpp -o main_cl.o

main_serial.o : ./serial/main.c ./host/src/sofm.h
	g++ -Wall -g -c ./serial/main.c -o main_serial.o

sofm.o : ./host/src/sofm.c
	g++ -Wall -g -c ./host/src/sofm.c -lm

AOCL_Utils.o : ./common/src/AOCL_Utils.cpp
	g++ -Wall -g -c ./common/src/AOCL_Utils.cpp

clean :
	rm -f *.o sofm_cl sofm_serial

