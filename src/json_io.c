#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json_io.h"

// Save in JSON
int save_network_json(const char *filename, Net *n) {
    FILE *f = fopen(filename, "w");
    if(!f){ fprintf(stderr, "Errore apertura %s\n", filename); return 0; }

    fprintf(f, "{\n");

    // ih weights
    fprintf(f, "  \"ih\": [\n");
    for(int i=0;i<n->hidden;i++){
        fprintf(f,"    [");
        for(int j=0;j<n->inputs;j++){
            fprintf(f,"%.17g", n->ih[i][j]);
            if(j<n->inputs-1) fprintf(f,", ");
        }
        fprintf(f, "]");
        if(i<n->hidden-1) fprintf(f,",");
        fprintf(f,"\n");
    }
    fprintf(f, "  ],\n");

    // ho weights
    fprintf(f, "  \"ho\": [\n");
    for(int i=0;i<n->outputs;i++){
        fprintf(f,"    [");
        for(int j=0;j<n->hidden;j++){
            fprintf(f,"%.17g", n->ho[i][j]);
            if(j<n->hidden-1) fprintf(f,", ");
        }
        fprintf(f, "]");
        if(i<n->outputs-1) fprintf(f,",");
        fprintf(f,"\n");
    }
    fprintf(f, "  ],\n");

    // bh bias hidden
    fprintf(f, "  \"bh\": [");
    for(int i=0;i<n->hidden;i++){
        fprintf(f,"%.17g", n->bh[i]);
        if(i<n->hidden-1) fprintf(f,", ");
    }
    fprintf(f, "],\n");

    // bo bias output
    fprintf(f, "  \"bo\": [");
    for(int i=0;i<n->outputs;i++){
        fprintf(f,"%.17g", n->bo[i]);
        if(i<n->outputs-1) fprintf(f,", ");
    }
    fprintf(f, "]\n");

    fprintf(f, "}\n");
    fclose(f);
    return 1;
}

// utility to skip spaces, tabs, newlines, commas
static void skip_chars(const char **p){
    while(**p && (**p==' '||**p=='\t'||**p=='\n'||**p=='\r'||**p==',')) (*p)++;
}

// Load from JSON
int load_network_json(const char *filename, Net *n) {
    FILE *f = fopen(filename,"r");
    if(!f) return 0;

    fseek(f,0,SEEK_END);
    long size = ftell(f);
    fseek(f,0,SEEK_SET);
    char *buf = malloc(size+1);
    if(!buf){ fclose(f); return 0; }
    fread(buf,1,size,f);
    buf[size]='\0';
    fclose(f);

    const char *p = buf;

    // ih
    const char *ph = strstr(p,"\"ih\"");
    if(!ph){ free(buf); return 0; }
    p = strchr(ph,'[')+1;
    for(int i=0;i<n->hidden;i++){
        p = strchr(p,'[')+1;
        for(int j=0;j<n->inputs;j++){
            skip_chars(&p);
            n->ih[i][j] = strtod(p,(char**)&p);
            skip_chars(&p);
        }
        p = strchr(p,']')+1;
    }

    // ho
    const char *pho = strstr(p,"\"ho\"");
    if(!pho){ free(buf); return 0; }
    p = strchr(pho,'[')+1;
    for(int i=0;i<n->outputs;i++){
        p = strchr(p,'[')+1;
        for(int j=0;j<n->hidden;j++){
            skip_chars(&p);
            n->ho[i][j] = strtod(p,(char**)&p);
            skip_chars(&p);
        }
        p = strchr(p,']')+1;
    }

    // bh
    const char *pbh = strstr(p,"\"bh\"");
    if(!pbh){ free(buf); return 0; }
    p = strchr(pbh,'[')+1;
    for(int i=0;i<n->hidden;i++){
        skip_chars(&p);
        n->bh[i] = strtod(p,(char**)&p);
        skip_chars(&p);
    }

    // bo
    const char *pbo = strstr(p,"\"bo\"");
    if(!pbo){ free(buf); return 0; }
    p = strchr(pbo,'[')+1;
    for(int i=0;i<n->outputs;i++){
        skip_chars(&p);
        n->bo[i] = strtod(p,(char**)&p);
        skip_chars(&p);
    }

    free(buf);
    return 1;
}
