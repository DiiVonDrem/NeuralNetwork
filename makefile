# Compilatore e flag
CC = gcc
CFLAGS = -Wall -O2 -lm

# Cartella sorgenti
SRC_DIR = src
SRC = $(SRC_DIR)/main.c \
      $(SRC_DIR)/network.c \
      $(SRC_DIR)/mathutils.c \
      $(SRC_DIR)/json_io.c
OBJ = $(SRC:.c=.o)

# Eseguibile
TARGET = neural_net

# Regole principali
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(OBJ) -o $(TARGET) $(CFLAGS)

# Compila ogni .c in .o
$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) -c $< -o $@

# Pulisci i file oggetto e l'eseguibile
clean:
ifeq ($(OS),Windows_NT)
	del /Q $(OBJ) $(TARGET).exe 2>nul
else
	rm -f $(OBJ) $(TARGET)
endif

# Esegui l'eseguibile
run: all
ifeq ($(OS),Windows_NT)
	$(TARGET).exe
else
	./$(TARGET)
endif
