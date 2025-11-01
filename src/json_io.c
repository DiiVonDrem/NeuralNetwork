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
    if(!f) {
        fprintf(stderr, "[WARN] File %s non trovato\n", filename);
        return 0;
    }

    fseek(f,0,SEEK_END);
    long size = ftell(f);
    fseek(f,0,SEEK_SET);
    char *buf = malloc(size+1);
    if(!buf){ fclose(f); fprintf(stderr,"[ERR] Memoria insufficiente\n"); return 0; }
    fread(buf,1,size,f);
    buf[size]='\0';
    fclose(f);

    const char *p = buf;

    // ih
    const char *ph = strstr(p,"\"ih\"");
    if(!ph){ free(buf); fprintf(stderr,"[WARN] 'ih' non trovato\n"); return 0; }
    const char *ptr = strchr(ph,'[');
    if(!ptr){ free(buf); fprintf(stderr,"[WARN] '[' non trovato dopo 'ih'\n"); return 0; }
    p = ptr+1;

    for(int i=0;i<n->hidden;i++){
        ptr = strchr(p,'[');
        if(!ptr){ free(buf); fprintf(stderr,"[WARN] '[' non trovato in ih row %d\n", i); return 0; }
        p = ptr+1;
        for(int j=0;j<n->inputs;j++){
            skip_chars(&p);
            char *end;
            n->ih[i][j] = strtod(p,&end);
            if(p == end){ free(buf); fprintf(stderr,"[WARN] Numero non valido in ih[%d][%d]\n",i,j); return 0; }
            p = end;
            skip_chars(&p);
        }
        ptr = strchr(p,']');
        if(!ptr){ free(buf); fprintf(stderr,"[WARN] ']' non trovato in ih row %d\n", i); return 0; }
        p = ptr+1;
    }

    // ho
    const char *pho = strstr(p,"\"ho\"");
    if(!pho){ free(buf); fprintf(stderr,"[WARN] 'ho' non trovato\n"); return 0; }
    ptr = strchr(pho,'[');
    if(!ptr){ free(buf); fprintf(stderr,"[WARN] '[' non trovato dopo 'ho'\n"); return 0; }
    p = ptr+1;

    for(int i=0;i<n->outputs;i++){
        ptr = strchr(p,'[');
        if(!ptr){ free(buf); fprintf(stderr,"[WARN] '[' non trovato in ho row %d\n", i); return 0; }
        p = ptr+1;
        for(int j=0;j<n->hidden;j++){
            skip_chars(&p);
            char *end;
            n->ho[i][j] = strtod(p,&end);
            if(p == end){ free(buf); fprintf(stderr,"[WARN] Numero non valido in ho[%d][%d]\n",i,j); return 0; }
            p = end;
            skip_chars(&p);
        }
        ptr = strchr(p,']');
        if(!ptr){ free(buf); fprintf(stderr,"[WARN] ']' non trovato in ho row %d\n", i); return 0; }
        p = ptr+1;
    }

    // bh
    const char *pbh = strstr(p,"\"bh\"");
    if(!pbh){ free(buf); fprintf(stderr,"[WARN] 'bh' non trovato\n"); return 0; }
    ptr = strchr(pbh,'[');
    if(!ptr){ free(buf); fprintf(stderr,"[WARN] '[' non trovato dopo 'bh'\n"); return 0; }
    p = ptr+1;
    for(int i=0;i<n->hidden;i++){
        skip_chars(&p);
        char *end;
        n->bh[i] = strtod(p,&end);
        if(p == end){ free(buf); fprintf(stderr,"[WARN] Numero non valido in bh[%d]\n",i); return 0; }
        p = end;
        skip_chars(&p);
    }

    // bo
    const char *pbo = strstr(p,"\"bo\"");
    if(!pbo){ free(buf); fprintf(stderr,"[WARN] 'bo' non trovato\n"); return 0; }
    ptr = strchr(pbo,'[');
    if(!ptr){ free(buf); fprintf(stderr,"[WARN] '[' non trovato dopo 'bo'\n"); return 0; }
    p = ptr+1;
    for(int i=0;i<n->outputs;i++){
        skip_chars(&p);
        char *end;
        n->bo[i] = strtod(p,&end);
        if(p == end){ free(buf); fprintf(stderr,"[WARN] Numero non valido in bo[%d]\n",i); return 0; }
        p = end;
        skip_chars(&p);
    }

    free(buf);
    return 1;
}