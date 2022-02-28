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

#include <unistd.h>
#include "input_utils.h"

#define STR2BOOL(X) ((X)[0] == 'T' || (X)[0] == 't' || (X)[0] == '1')

void append_token_to_CNN(int cnn_or_bach, int layer, int col, char *token, cnn_t *cnn) {
    if (cnn_or_bach == CNN_TYPE)
        switch (col) {
            case 0:
                cnn[layer].layer = atoi(token);
                break;
            case 1: //Kn
                cnn[layer].kmin = atoi(token);
                cnn[layer].kmax = atoi(token);
                cnn[layer].kstep = 1;
                break;
            case 2: //Wo
                break;
            case 3: //Ho
                break;
            case 4: //t
                cnn[layer].nmin = atoi(token);
                cnn[layer].nmax = atoi(token);
                cnn[layer].nstep = 1;
                break;
            case 5: //Kh
                cnn[layer].rmin = atoi(token);
                cnn[layer].rmax = atoi(token);
                cnn[layer].rstep = 1;
                break;
            case 6: //Kw
                cnn[layer].smin = atoi(token);
                cnn[layer].smax = atoi(token);
                cnn[layer].sstep = 1;
                break;
            case 7: //Ci
                cnn[layer].cmin = atoi(token);
                cnn[layer].cmax = atoi(token);
                cnn[layer].cstep = 1;
                break;
            case 8: //Wi
                cnn[layer].wmin = atoi(token);
                cnn[layer].wmax = atoi(token);
                cnn[layer].wstep = 1;
                break;
            case 9: //Hi
                cnn[layer].hmin = atoi(token);
                cnn[layer].hmax = atoi(token);
                cnn[layer].hstep = 1;
                break;
        }
    else
        switch (col) {
            case 0:
                cnn[layer].nmin = atoi(token);
                break;
            case 1:
                cnn[layer].nmax = atoi(token);
                break;
            case 2:
                cnn[layer].nstep = atoi(token);
                break;
            case 3:
                cnn[layer].kmin = atoi(token);
                break;
            case 4:
                cnn[layer].kmax = atoi(token);
                break;
            case 5:
                cnn[layer].kstep = atoi(token);
                break;
            case 6:
                cnn[layer].cmin = atoi(token);
                break;
            case 7:
                cnn[layer].cmax = atoi(token);
                break;
            case 8:
                cnn[layer].cstep = atoi(token);
                break;
            case 9:
                cnn[layer].hmin = atoi(token);
                break;
            case 10:
                cnn[layer].hmax = atoi(token);
                break;
            case 11:
                cnn[layer].hstep = atoi(token);
                break;
            case 12:
                cnn[layer].wmin = atoi(token);
                break;
            case 13:
                cnn[layer].wmax = atoi(token);
                break;
            case 14:
                cnn[layer].wstep = atoi(token);
                break;
            case 15:
                cnn[layer].rmin = atoi(token);
                break;
            case 16:
                cnn[layer].rmax = atoi(token);
                break;
            case 17:
                cnn[layer].rstep = atoi(token);
                break;
            case 18:
                cnn[layer].smin = atoi(token);
                break;
            case 19:
                cnn[layer].smax = atoi(token);
                break;
            case 20:
                cnn[layer].sstep = atoi(token);
                break;
        }
}

/**
 * Processes the command line options and returns an evalConfig instance.
 *
 * @param argv  An array with the command name and its arguments
 * @return An evalConfig instance
 */
evalConfig_t *get_evalConfig(char *argv[]) {

#define ARG_ALGORITHM_NAME      argv[1]
#define ARG_CONFIG_FILENAME     argv[2]
#define ARG_TYPE_CNN_OR_BATCH   argv[3]
#define ARG_TMIN                argv[4]
#define ARG_TEST                argv[5]
#define ARG_DEBUG               argv[6]
#define ARG_OUTPUT_FILENAME     argv[7]

    FILE *fd_conf = fopen(ARG_CONFIG_FILENAME, "r"); // open config file
    char *line = NULL;
    size_t len = 0;
    ssize_t read;
    const char delimiter[] = "\t";
    char *token;
    char *line_str;
    int col;
    evalConfig_t *ec = (evalConfig_t *) malloc(sizeof(evalConfig_t));
    char format_str[4];

    strncpy(ec->algorithm_name, ARG_ALGORITHM_NAME, 128);
    ec->type = (strncmp(ARG_TYPE_CNN_OR_BATCH, "cnn", 3) == 0) ? CNN_TYPE : BATCH_TYPE;
    ec->tmin = atof(ARG_TMIN);
    ec->test = STR2BOOL(ARG_TEST);
    ec->debug = STR2BOOL(ARG_DEBUG);

    if (! ec->test) {
        if( access( ARG_OUTPUT_FILENAME, F_OK ) == 0 ) {
            printf("Output file '%s' already exits.\n"
                   "Please, delete it if you want to rerun this evaluation.\n",
                   ARG_OUTPUT_FILENAME);
            exit(EXIT_SUCCESS);
        }
        ec->fd_out = fopen(ARG_OUTPUT_FILENAME, "w");
    } else {
        ec->fd_out = NULL;
    }

    ec->tensor_format = convdirect_get_tensor_format(ARG_ALGORITHM_NAME);
    if (ec->tensor_format == nchw)
        strncpy(format_str, "NCHW", 4);
    else
        strncpy(format_str, "NHWC", 4);

    if (ec->debug && !ec->test) {
        printf("Warning: As mode debug is enabled, test mode has been also enabled.\n");
        ec->test = true;
        ec->tmin = 0.0;
    }

    if (ec->test && (ec->tmin > 0.0)) {
        printf("Error: If 'Test check' is enabled then 'Tmin' must be 0. Please, fix it and run again.\n");
        exit(EXIT_FAILURE);
    }

    printf("\n");
    //                   10        20        30        40        50        60        70        80
    //          01234567890123456789012345678901234567890123456789012345678901234567890123456789
    char l[] = "================================================================================"
               "==============================================";
    printf(" %s\n", l); // -----------------------------------------------------------------
    printf(" |%s%*sE V A L    C O N F I G U R A T I O N%*s%s|\n", COLOR_BOLDYELLOW, 44, " ", 44, " ", COLOR_RESET);
    printf(" %s\n", l); // -----------------------------------------------------------------
    printf(" | Algorithm selected     : %s%-98s%s|\n", COLOR_BOLDYELLOW, ARG_ALGORITHM_NAME, COLOR_RESET);
    printf(" | Configuration selected : %-98s|\n", ARG_CONFIG_FILENAME);
    printf(" | Test verification      : %-98s|\n", ec->test ? "ON" : "OFF");
    printf(" | Debug mode             : %-98s|\n", ec->debug ? "ON" : "OFF");
    printf(" | Results file           : %-98s|\n", ARG_OUTPUT_FILENAME);
    printf(" %s\n", l); // -----------------------------------------------------------------

    ec->cnn_layers = 0;
    while (getline(&line, &len, fd_conf) != -1)
        if (line[0] != '#') {
            for (col = 0, line_str = line;; col++, line_str = NULL) {
                token = strtok(line_str, delimiter);
                if (token == NULL)
                    break;
                append_token_to_CNN(ec->type, ec->cnn_layers, col, token, ec->cnn);
            }
            ec->cnn_layers++;
        }
    fclose(fd_conf);
    return ec;
}

void free_evalConfig(evalConfig_t *evalConfig) {
    free(evalConfig);
}

//printf("nmin=%d; nmax=%d; nstep=%d; kmin=%d; kmax=%d; kstep=%d; cmin=%d; cmax=%d; cstep=%d; hmin=%d; hmax=%d; hstep=%d, wmin=%d; wmax=%d; wstep=%d; rmin=%d; rmax=%d; rstep=%d; smin=%d; smax=%d; sstep=%d\n",
//       cnn[*cnn_num].nmin, cnn[*cnn_num].nmax, cnn[*cnn_num].nstep,
//       cnn[*cnn_num].kmin, cnn[*cnn_num].kmax, cnn[*cnn_num].kstep,
//       cnn[*cnn_num].cmin, cnn[*cnn_num].cmax, cnn[*cnn_num].cstep,
//       cnn[*cnn_num].hmin, cnn[*cnn_num].hmax, cnn[*cnn_num].hstep,
//      cnn[*cnn_num].wmin, cnn[*cnn_num].wmax, cnn[*cnn_num].wstep,
//       cnn[*cnn_num].rmin, cnn[*cnn_num].rmax, cnn[*cnn_num].rstep,
//       cnn[*cnn_num].smin, cnn[*cnn_num].smax, cnn[*cnn_num].sstep);
