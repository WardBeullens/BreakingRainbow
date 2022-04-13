#ifndef __OPTIONS_H__
#define __OPTIONS_H__


#define OP_SYS_READ  128
#define OP_SYS_WRITE 129

#define OP_BW1_READ  130
#define OP_BW1_WRITE 131

#define OP_BM_READ   132
#define OP_BM_WRITE  133

#define OP_SEED      136
#define OP_ITERATION 137

#define OP_IT_READ   138
#define OP_IT_WRITE  139

#define OP_BW1_RUN   140
#define OP_BM_RUN    141
#define OP_BW3_RUN   142
#define OP_ALL_RUN   143

#define OP_FINAL_BW  144

#define OP_BIND      145

#define OP_CHALLENGE 146

class Options
{
    public:

    char *system_file;
    int system;

    char *challenge_file;
    int challenge;

    char *bw1_file;
    int bw1;
    bool bw1_run;

    char *bm_file;
    int bm;
    bool bm_run;

    bool bw3_run;

    uint32_t seed;

    int iteration;

    bool all_it;
    bool final_it;

    char *it_read_file;
    char *it_write_file;

    bool it_read;
    bool it_write;

    int *bind;

    Options()
    {
       system_file = NULL;
       bw1_file = NULL;
       bm_file = NULL;
       it_read_file = NULL;
       it_write_file = NULL;
       bind = NULL;
       challenge_file = NULL;
    }

    ~Options()
    {
       if (system_file != NULL) free(system_file);
       if (bw1_file != NULL) free(bw1_file);
       if (bm_file != NULL) free(bm_file);
       if (it_read_file != NULL) free(it_read_file);
       if (it_write_file != NULL) free(it_write_file);
       if (bind != NULL) free(bind);
       if (challenge_file != NULL) free(challenge_file);
    }

    void help(char * argv[]);

    void parse(int argc, char * argv[]);
};

#endif // ifndef __OPTIONS_H__

