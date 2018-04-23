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

sofm_cl : main.o AOCL_Utils.o
	g++ -Wall -g main.o AOCL_Utils.o -o sofm_cl -framework OpenCL

main.o : ./host/src/main.cpp ./common/inc/AOCL_Utils.h
	g++ -Wall -g -c ./host/src/main.cpp

# hpot_support.o : hpot_support.c datatypes.h list.h hpot_support.h
# 	gcc -Wall -g -c hpot_support.c

# lab2.o : lab2.c datatypes.h list.h hpot_support.h
# 	gcc -Wall -g -c lab2.c

# testlist : testlist.o list.o hpot_support.o
# 	gcc -Wall -g list.o hpot_support.o testlist.o -o testlist

AOCL_Utils.o : ./common/src/AOCL_Utils.cpp
	g++ -Wall -g -c ./common/src/AOCL_Utils.cpp

clean :
	rm -f *.o sofm_cl 

