CC = nvcc
LIB_HOME = src
INCLUDE = -I$(LIB_HOME)
OPT = -std=c++14
OPT += -O2
OPT += -lcublas
OPT += -code=sm_80
OPT += -arch=compute_80
# OPT += -rdc=true

ifdef PRETTYPRINT
OPT += -DPRETTYPRINT
endif

ifdef DTYPE
OPT += -DDTYPE=$(DTYPE)
endif

ifdef TILE_DIM
OPT += -DTILE_DIM=$(TILE_DIM)
endif

ifdef BLOCK_ROWS
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
TARGET := $(TARGETDIR)/main_executable

# Main source file and other sources in the src directory
MAIN_SOURCE := main.cu
SOURCES := $(wildcard $(LIB_HOME)/*.cu)
OBJECTS := $(BUILDDIR)/main.o $(patsubst $(LIB_HOME)/%.cu, $(BUILDDIR)/%.o, $(SOURCES))

all: $(TARGET)

debug: OPT += -g
debug: OPT += -DDEBUG
debug: all

$(TARGET): $(OBJECTS)
	@mkdir -p $(TARGETDIR)
	$(CC) $(OBJECTS) -o $@ $(INCLUDE) $(OPT)

$(BUILDDIR)/%.o: $(LIB_HOME)/%.cu
	@mkdir -p $(BUILDDIR)
	$(CC) -c $< -o $@ $(INCLUDE) $(OPT)

# Special rule for compiling the main.cu which is not in src
$(BUILDDIR)/main.o: $(MAIN_SOURCE)
	@mkdir -p $(BUILDDIR)
	$(CC) -c $< -o $@ $(INCLUDE) $(OPT)

clean:
	rm -rf $(BUILDDIR) $(TARGETDIR)
