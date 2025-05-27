CXX = g++
CXXFLAGS = -O3 -Wall amx.cpp -std=c++20 -mamx-tile -mamx-bf16 -I.

# Target executable
TARGET = amx

# Source files
SOURCES = amx.cpp

# Default target
all: $(TARGET)

# Linking stage: create executable from object files
$(TARGET):
	$(CXX) $(CXXFLAGS) -o $(TARGET)

# Clean up generated files
clean:
	rm -f $(TARGET)
