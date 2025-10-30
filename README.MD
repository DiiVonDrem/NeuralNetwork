This is a working, simple, Neural Network, all coded in C.


The MAKEFILE is an easy way to compile the files in .exe files and run them.


Under src/ are all the .c and .h files:

    -mathutils.c : Contains Math functions to train the NN;

    -mathutils.h : Header for "mathutils.c";

    -network.c   : Contains functions to train the NN

    -network.h   : Header for "network.c"

    -json_io.c   : Contains functions to save/load data in a .json file

    -json_io.h   : Header for "json_io.c"

    -main.c      : Used to run the NN, either training ther NN if there are no data in json files or loads data from them


Under data/ there are the training data saved in .json files
