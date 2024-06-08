CC = nvcc
LIB_HOME = src
INCLUDE = -I$(LIB_HOME)
OPT = -std=c++14

ifdef DTYPE
OPT += -DDTYPE=$(DTYPE)
endif

ifdef TILE_DIM
OPT += -DTILE_DIM=$(TILE_DIM)
endif

ifdef TILE_DIM
OPT += -DBLOCK_ROWS=$(BLOCK_ROWS)
endif

ifdef LOG
OPT += -DLOG
endif

ifdef DEBUG
OPT += -g
endif

#################

BUILDDIR := obj
TARGETDIR := bin

# List of CUDA source files
SOURCES := transpose.cu 

# List of corresponding binaries
BINARIES := $(patsubst %.cu, $(TARGETDIR)/%, $(SOURCES))

all: $(BINARIES)

debug: OPT += -g
debug: all

$(TARGETDIR)/%: %.cu
	@mkdir -p $(@D)
	$(CC) $< -o $@ $(INCLUDE) $(OPT)

clean:
	rm -rf $(BUILDDIR) $(TARGETDIR)
