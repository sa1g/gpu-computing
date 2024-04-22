CC = c++
LIB_HOME = src
LIBS = -Lobj
INCLUDE = -Isrc
OPT = -std=c++14

MAIN = transpose.cc

#################
# args:
#	 DTYPE = DTYPE:=[float, double] 
#	 BLOCK = BLOCK:=[BLOCK] (it just needs to be defined) -> if you either want to use block method or not
#	 DEBUG = DEBUG:=[DEBUG] (it just needs to be defined)
#################

# Check if DTYPE is set and append -DDTYPE=<type> if so
ifdef DTYPE
OPT += -DDTYPE=$(DTYPE)
endif

ifdef BLOCK
OPT += -DBLOCK
endif

ifdef DEBUG
OPT += -g
endif

#################

BUILDDIR := obj
TARGETDIR := bin

all: $(TARGETDIR)/transpose

O0: OPT += -O0
O0: all

O1: OPT += -O1
O1: all

O2: OPT += -O2 #-march=native -ftree-vectorize
O2: all
 
O3: OPT += -O3 #-march=native -ftree-vectorize
O3: all

debug: OPT += -g
debug: all

OBJECTS := $(BUILDDIR)/matrix.o \
           $(BUILDDIR)/utils.o \
# 		   $(BUILDDIR)/t.o 
		   


$(TARGETDIR)/transpose: $(MAIN) $(OBJECTS)
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(LIBS) $(OPT)

$(BUILDDIR)/%.o: $(LIB_HOME)/%.cc $(LIB_HOME)/%.h
	mkdir -p $(BUILDDIR) $(TARGETDIR)
	$(CC) -c -o $@ $(INCLUDE) $< $(OPT)

clean:
	rm -f $(BUILDDIR)/*.o $(TARGETDIR)/transpose
	# This is not needed, but just in case
	rm -rf $(BUILDDIR)
	rm -rf $(TARGETDIR)
