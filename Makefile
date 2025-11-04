CC := g++

# Directories and Files
SRCDIR := src
BINDIR := bin
SOURCE := $(SRCDIR)/main.cpp
TARGET := $(BINDIR)/main.out


# Set OpenCV flags
PKG_FLAGS := $(shell pkg-config --cflags --libs opencv4)

# C++ flags
CXXFLAGS := -Wall -Wextra -Wconversion -O2 -std=c++20 $(PKG_FLAGS)

# Default target
.PHONY: all
all: $(TARGET)

# Compiles the source file into the target, ensuring
# that the bin directory is there.
$(TARGET): $(SOURCE) | $(BINDIR)
	@echo "Compiling $(SOURCE) -> $(TARGET)"
	$(CC) $(SOURCE) -o $(TARGET) $(CXXFLAGS)

$(BINDIR):
	@echo "Creating directory $(BINDIR)..."
	mkdir -p $(BINDIR)

# Rule to clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(BINDIR)