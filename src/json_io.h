#ifndef JSON_IO_H
#define JSON_IO_H

#include "network.h"

// Save in JSON
int save_network_json(const char *filename, Net *n);

// Load from JSON
int load_network_json(const char *filename, Net *n);

#endif
