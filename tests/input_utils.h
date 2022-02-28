/**
 * This file is part of convDirect
 *
 * Copyright (C) 2021-22 Universitat Politècnica de València and
 *                       Universitat Jaume I
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "colors.h"
#include "convdirect.h"

#define CNN_MAX_LAYERS 128
#define CNN_TYPE     0
#define BATCH_TYPE   1

typedef struct cnn {
    int nmin;
    int nmax;
    int nstep;

    int kmin;
    int kmax;
    int kstep;

    int cmin;
    int cmax;
    int cstep;

    int hmin;
    int hmax;
    int hstep;

    int wmin;
    int wmax;
    int wstep;

    int rmin;
    int rmax;
    int rstep;

    int smin;
    int smax;
    int sstep;

    int layer;
} cnn_t;

typedef struct evalConfig {
    char algorithm_name[128];
    cnn_t cnn[CNN_MAX_LAYERS];
    int cnn_layers;
    double tmin;
    unsigned char type;
    bool test;
    bool debug;
    FILE *fd_out;
    tensor_format_t tensor_format;
} evalConfig_t;


void append_token_to_CNN(int cnn_or_bach, int layer, int col, char *token, cnn_t *cnn);

evalConfig_t *get_evalConfig(char **);

void free_evalConfig(evalConfig_t *);
