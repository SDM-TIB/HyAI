CC=		cc

CFLAGS=		-Wall -Wextra -O3 -ffast-math -std=gnu99 -march=native -mtune=native -Wshadow -Wpointer-arith -Wstrict-prototypes -Wmissing-prototypes
DFLAGS=

#CFLAGS=		-Wall -Wextra -O0 -g -std=gnu99
#DFLAGS=		-DPRGDEBUG

PROG=		semEP-node

#SOLVER=		util.c graph_adj.c hash_map.c hash_iset.c input.c semEP-node.c main.c
SOLVER=		util.c graph_adj.c hash_iset.c hash_map.c input.c solver.c main.c
LIBS=		-lm
SOLVEROBJS=	$(SOLVER:.c=.o)

INSTALLDIR=	../

.SUFFIXES:.c .o

all:		$(PROG)

$(PROG):	$(SOLVEROBJS)
		$(CC) $(CFLAGS) -o $(INSTALLDIR)$(PROG) $(SOLVEROBJS) $(LIBS)

.c.o:
	$(CC) -c  $(CFLAGS) $(DFLAGS) $< -o $@

.PHONY : clean

clean :
	rm -rf $(INSTALLDIR)$(PROG) *.o *.dSYM $(INSTALLDIR)*.dSYM *~

