/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
// openmp
#include <omp.h>

// example specific

#include "ocp_problem_model/ocp_problem_model.h"


#include "ocp_problem_p_global_precompute_fun.h"
#include "ocp_problem_cost/ocp_problem_cost.h"



#include "acados_solver_ocp_problem.h"

#define NX     OCP_PROBLEM_NX
#define NZ     OCP_PROBLEM_NZ
#define NU     OCP_PROBLEM_NU
#define NP     OCP_PROBLEM_NP
#define NP_GLOBAL     OCP_PROBLEM_NP_GLOBAL
#define NY0    OCP_PROBLEM_NY0
#define NY     OCP_PROBLEM_NY
#define NYN    OCP_PROBLEM_NYN

#define NBX    OCP_PROBLEM_NBX
#define NBX0   OCP_PROBLEM_NBX0
#define NBU    OCP_PROBLEM_NBU
#define NG     OCP_PROBLEM_NG
#define NBXN   OCP_PROBLEM_NBXN
#define NGN    OCP_PROBLEM_NGN

#define NH     OCP_PROBLEM_NH
#define NHN    OCP_PROBLEM_NHN
#define NH0    OCP_PROBLEM_NH0
#define NPHI   OCP_PROBLEM_NPHI
#define NPHIN  OCP_PROBLEM_NPHIN
#define NPHI0  OCP_PROBLEM_NPHI0
#define NR     OCP_PROBLEM_NR

#define NS     OCP_PROBLEM_NS
#define NS0    OCP_PROBLEM_NS0
#define NSN    OCP_PROBLEM_NSN

#define NSBX   OCP_PROBLEM_NSBX
#define NSBU   OCP_PROBLEM_NSBU
#define NSH0   OCP_PROBLEM_NSH0
#define NSH    OCP_PROBLEM_NSH
#define NSHN   OCP_PROBLEM_NSHN
#define NSG    OCP_PROBLEM_NSG
#define NSPHI0 OCP_PROBLEM_NSPHI0
#define NSPHI  OCP_PROBLEM_NSPHI
#define NSPHIN OCP_PROBLEM_NSPHIN
#define NSGN   OCP_PROBLEM_NSGN
#define NSBXN  OCP_PROBLEM_NSBXN



// ** solver data **

ocp_problem_solver_capsule * ocp_problem_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(ocp_problem_solver_capsule));
    ocp_problem_solver_capsule *capsule = (ocp_problem_solver_capsule *) capsule_mem;

    return capsule;
}


int ocp_problem_acados_free_capsule(ocp_problem_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int ocp_problem_acados_create(ocp_problem_solver_capsule* capsule)
{
    int N_shooting_intervals = OCP_PROBLEM_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return ocp_problem_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int ocp_problem_acados_update_time_steps(ocp_problem_solver_capsule* capsule, int N, double* new_time_steps)
{

    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "ocp_problem_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;

}

/**
 * Internal function for ocp_problem_acados_create: step 1
 */
void ocp_problem_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->nlp_cost[0] = EXTERNAL;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = EXTERNAL;

    nlp_solver_plan->nlp_cost[N] = LINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = DISCRETE_MODEL;
        // discrete dynamics does not need sim solver option, this field is ignored
        nlp_solver_plan->sim_solver_plan[i].sim_solver = INVALID_SIM_SOLVER;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = FIXED_STEP;
}


static ocp_nlp_dims* ocp_problem_acados_create_setup_dimensions(ocp_problem_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    
    nbxe[0] = 22;
    
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 702);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 1352);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);
    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for ocp_problem_acados_create: step 3
 */
void ocp_problem_acados_create_setup_functions(ocp_problem_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    // NOTE: p_global_precompute_fun cannot use external_workspace!!!
    ext_fun_opts.external_workspace = false;
    capsule->p_global_precompute_fun.casadi_fun = &ocp_problem_p_global_precompute_fun;
    capsule->p_global_precompute_fun.casadi_work = &ocp_problem_p_global_precompute_fun_work;
    capsule->p_global_precompute_fun.casadi_sparsity_in = &ocp_problem_p_global_precompute_fun_sparsity_in;
    capsule->p_global_precompute_fun.casadi_sparsity_out = &ocp_problem_p_global_precompute_fun_sparsity_out;
    capsule->p_global_precompute_fun.casadi_n_in = &ocp_problem_p_global_precompute_fun_n_in;
    capsule->p_global_precompute_fun.casadi_n_out = &ocp_problem_p_global_precompute_fun_n_out;
    external_function_casadi_create(&capsule->p_global_precompute_fun, &ext_fun_opts);
    // asserts
    if (capsule->p_global_precompute_fun.in_num != 1)
    {
        printf("input dimension of p_global_precompute_fun should have 1 input, got %d\n", capsule->p_global_precompute_fun.in_num);
        exit(1);
    }
    if (capsule->p_global_precompute_fun.out_num != 1)
    {
        printf("input dimension of p_global_precompute_fun should have 1 output, got %d\n", capsule->p_global_precompute_fun.out_num);
        exit(1);
    }
    if (capsule->p_global_precompute_fun.args_size[0] != 702)
    {
        printf("input dimension of p_global_precompute_fun should be np_global = 702, got %d\n", capsule->p_global_precompute_fun.args_size[0]);
        exit(1);
    }
    if (capsule->p_global_precompute_fun.res_size[0] != 1352)
    {
        printf("output dimension of p_global_precompute_fun should be n_global_data = 1352, got %d\n", capsule->p_global_precompute_fun.res_size[0]);
        exit(1);
    }

    ext_fun_opts.with_global_data = true;
    ext_fun_opts.external_workspace = true;
    // external cost
    MAP_CASADI_FNC(ext_cost_0_fun, ocp_problem_cost_ext_cost_0_fun);
    MAP_CASADI_FNC(ext_cost_0_fun_jac, ocp_problem_cost_ext_cost_0_fun_jac);
    MAP_CASADI_FNC(ext_cost_0_fun_jac_hess, ocp_problem_cost_ext_cost_0_fun_jac_hess);
    MAP_CASADI_FNC(ext_cost_0_hess_xu_p, ocp_problem_cost_ext_cost_0_hess_xu_p);




    // discrete dynamics
    capsule->discr_dyn_phi_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun[i], ocp_problem_dyn_disc_phi_fun);
    }

    capsule->discr_dyn_phi_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt[i], ocp_problem_dyn_disc_phi_fun_jac);
    }

  
    capsule->discr_dyn_phi_jac_p_hess_xu_p = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_jac_p_hess_xu_p[i], ocp_problem_dyn_disc_phi_jac_p_hess_xu_p);
    }
  

  
    capsule->discr_dyn_phi_fun_jac_ut_xt_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        MAP_CASADI_FNC(discr_dyn_phi_fun_jac_ut_xt_hess[i], ocp_problem_dyn_disc_phi_fun_jac_hess);
    }
    // external cost
    capsule->ext_cost_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(ext_cost_fun[i], ocp_problem_cost_ext_cost_fun);
    }

    capsule->ext_cost_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(ext_cost_fun_jac[i], ocp_problem_cost_ext_cost_fun_jac);
    }

    capsule->ext_cost_fun_jac_hess = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(ext_cost_fun_jac_hess[i], ocp_problem_cost_ext_cost_fun_jac_hess);
    }

    
    capsule->ext_cost_hess_xu_p = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(ext_cost_hess_xu_p[i], ocp_problem_cost_ext_cost_hess_xu_p);
    }

    

#undef MAP_CASADI_FNC
}


/**
 * Internal function for ocp_problem_acados_create: step 5
 */
void ocp_problem_acados_create_set_default_parameters(ocp_problem_solver_capsule* capsule)
{

    // no parameters defined


    // initialize global parameters to nominal value
    double* p_global = calloc(NP_GLOBAL, sizeof(double));

    ocp_problem_acados_set_p_global_and_precompute_dependencies(capsule, p_global, NP_GLOBAL);

    free(p_global);
}


/**
 * Internal function for ocp_problem_acados_create: step 5
 */
void ocp_problem_acados_setup_nlp_in(ocp_problem_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = capsule->nlp_in;
    /************************************************
    *  nlp_out
    ************************************************/
    ocp_nlp_out * nlp_out = capsule->nlp_out;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        ocp_problem_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    double time_step = 0.016;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.016;
        cost_scaling[1] = 0.016;
        cost_scaling[2] = 0.016;
        cost_scaling[3] = 0.016;
        cost_scaling[4] = 0.016;
        cost_scaling[5] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }



    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun", &capsule->discr_dyn_phi_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_phi_jac_p_hess_xu_p",
                                   &capsule->discr_dyn_phi_jac_p_hess_xu_p[i]);
        
        
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "disc_dyn_fun_jac_hess",
                                   &capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i]);
    }

    /**** Cost ****/
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun", &capsule->ext_cost_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac", &capsule->ext_cost_0_fun_jac);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_fun_jac_hess", &capsule->ext_cost_0_fun_jac_hess);
    
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "ext_cost_hess_xu_p", &capsule->ext_cost_0_hess_xu_p);
    
    
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun", &capsule->ext_cost_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac", &capsule->ext_cost_fun_jac[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac_hess", &capsule->ext_cost_fun_jac_hess[i-1]);
        
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "ext_cost_hess_xu_p", &capsule->ext_cost_hess_xu_p[i-1]);
        
        
    }







    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;
    idxbx0[13] = 13;
    idxbx0[14] = 14;
    idxbx0[15] = 15;
    idxbx0[16] = 16;
    idxbx0[17] = 17;
    idxbx0[18] = 18;
    idxbx0[19] = 19;
    idxbx0[20] = 20;
    idxbx0[21] = 21;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    lbx0[0] = -0.7687100768089294;
    ubx0[0] = -0.7687100768089294;
    lbx0[1] = 0.4806603193283081;
    ubx0[1] = 0.4806603193283081;
    lbx0[2] = -0.5215718746185303;
    ubx0[2] = -0.5215718746185303;
    lbx0[3] = -0.6394777297973633;
    ubx0[3] = -0.6394777297973633;
    lbx0[4] = -0.5648282170295715;
    ubx0[4] = -0.5648282170295715;
    lbx0[5] = 0.9999998807907104;
    ubx0[5] = 0.9999998807907104;
    lbx0[6] = 0.9999998807907104;
    ubx0[6] = 0.9999998807907104;
    lbx0[7] = 0.9999998807907104;
    ubx0[7] = 0.9999998807907104;
    lbx0[8] = 0.9999998807907104;
    ubx0[8] = 0.9999998807907104;
    lbx0[9] = -5.3344268798828125;
    ubx0[9] = -5.3344268798828125;
    lbx0[10] = 8.898691177368164;
    ubx0[10] = 8.898691177368164;
    lbx0[11] = 1.6356430053710938;
    ubx0[11] = 1.6356430053710938;
    lbx0[12] = -1.3464868068695068;
    ubx0[12] = -1.3464868068695068;
    lbx0[14] = 0.5;
    ubx0[14] = 0.5;
    lbx0[15] = -0.25;
    ubx0[15] = -0.25;
    lbx0[16] = -0.7864723205566406;
    ubx0[16] = -0.7864723205566406;
    lbx0[17] = 2.8218162059783936;
    ubx0[17] = 2.8218162059783936;
    lbx0[18] = 0.4662337899208069;
    ubx0[18] = 0.4662337899208069;
    lbx0[19] = 0.9999998807907104;
    ubx0[19] = 0.9999998807907104;
    lbx0[20] = 0.9999998807907104;
    ubx0[20] = 0.9999998807907104;
    lbx0[21] = -4.839761734008789;
    ubx0[21] = -4.839761734008789;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(22 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    idxbxe_0[13] = 13;
    idxbxe_0[14] = 14;
    idxbxe_0[15] = 15;
    idxbxe_0[16] = 16;
    idxbxe_0[17] = 17;
    idxbxe_0[18] = 18;
    idxbxe_0[19] = 19;
    idxbxe_0[20] = 20;
    idxbxe_0[21] = 21;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);








    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = 2;
    ubu[0] = 20;
    lbu[1] = -6;
    ubu[1] = 6;
    lbu[2] = -6;
    ubu[2] = 6;
    lbu[3] = -6;
    ubu[3] = 6;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);















    /* terminal constraints */













}


static void ocp_problem_acados_create_set_opts(ocp_problem_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/


    int nlp_solver_exact_hessian = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess", &nlp_solver_exact_hessian);

    int exact_hess_dyn = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_dyn", &exact_hess_dyn);

    int exact_hess_cost = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_cost", &exact_hess_cost);

    int exact_hess_constr = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "exact_hess_constr", &exact_hess_constr);

    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);

    double globalization_fixed_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_fixed_step_length", &globalization_fixed_step_length);




    int with_solution_sens_wrt_params = true;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.0000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 5;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    int log_primal_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_primal_step_norm", &log_primal_step_norm);

    int log_dual_step_norm = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "log_dual_step_norm", &log_dual_step_norm);

    double nlp_solver_tol_min_step_norm = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_min_step_norm", &nlp_solver_tol_min_step_norm);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "BALANCE");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    // set SQP specific options
    double nlp_solver_tol_stat = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 100;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "max_iter", &nlp_solver_max_iter);

    // set options for adaptive Levenberg-Marquardt Update
    bool with_adaptive_levenberg_marquardt = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "with_adaptive_levenberg_marquardt", &with_adaptive_levenberg_marquardt);

    double adaptive_levenberg_marquardt_lam = 5;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_lam", &adaptive_levenberg_marquardt_lam);

    double adaptive_levenberg_marquardt_mu_min = 0.0000000000000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu_min", &adaptive_levenberg_marquardt_mu_min);

    double adaptive_levenberg_marquardt_mu0 = 0.001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "adaptive_levenberg_marquardt_mu0", &adaptive_levenberg_marquardt_mu0);

    bool eval_residual_at_max_iter = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "eval_residual_at_max_iter", &eval_residual_at_max_iter);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);



    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "cost_numerical_hessian", &ext_cost_num_hess);
    }
}


/**
 * Internal function for ocp_problem_acados_create: step 7
 */
void ocp_problem_acados_set_nlp_out(ocp_problem_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    x0[0] = -0.7687100768089294;
    x0[1] = 0.4806603193283081;
    x0[2] = -0.5215718746185303;
    x0[3] = -0.6394777297973633;
    x0[4] = -0.5648282170295715;
    x0[5] = 0.9999998807907104;
    x0[6] = 0.9999998807907104;
    x0[7] = 0.9999998807907104;
    x0[8] = 0.9999998807907104;
    x0[9] = -5.3344268798828125;
    x0[10] = 8.898691177368164;
    x0[11] = 1.6356430053710938;
    x0[12] = -1.3464868068695068;
    x0[14] = 0.5;
    x0[15] = -0.25;
    x0[16] = -0.7864723205566406;
    x0[17] = 2.8218162059783936;
    x0[18] = 0.4662337899208069;
    x0[19] = 0.9999998807907104;
    x0[20] = 0.9999998807907104;
    x0[21] = -4.839761734008789;


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for ocp_problem_acados_create: step 9
 */
int ocp_problem_acados_create_precompute(ocp_problem_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int ocp_problem_acados_create_with_discretization(ocp_problem_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != OCP_PROBLEM_N && !new_time_steps) {
        fprintf(stderr, "ocp_problem_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, OCP_PROBLEM_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    ocp_problem_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = ocp_problem_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    ocp_problem_acados_create_set_opts(capsule);

    // 4) create and set nlp_out
    // 4.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 4.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    ocp_problem_acados_set_nlp_out(capsule);

    // 5) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 6) setup functions, nlp_in and default parameters
    ocp_problem_acados_create_setup_functions(capsule);
    ocp_problem_acados_setup_nlp_in(capsule, N, new_time_steps);
    ocp_problem_acados_create_set_default_parameters(capsule);

    // 7) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);


    // 8) do precomputations
    int status = ocp_problem_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int ocp_problem_acados_update_qp_solver_cond_N(ocp_problem_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from ocp_problem_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = ocp_problem_acados_create_precompute(capsule);
    return status;
}


int ocp_problem_acados_reset(ocp_problem_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "pi", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int ocp_problem_acados_update_params(ocp_problem_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 0;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int ocp_problem_acados_update_params_sparse(ocp_problem_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int ocp_problem_acados_set_p_global_and_precompute_dependencies(ocp_problem_solver_capsule* capsule, double* data, int data_len)
{

    external_function_casadi* fun = &capsule->p_global_precompute_fun;
    fun->args[0] = data;
    int np_global = 702;

    if (data_len != np_global)
    {
        printf("ocp_problem_acados_set_p_global_and_precompute_dependencies: np_global = %d should match data_len = %d. Exiting.\n", np_global, data_len);
        exit(1);
    }

    ocp_nlp_in *in = ocp_problem_acados_get_nlp_in(capsule);
    fun->res[0] = in->global_data;

    fun->casadi_fun((const double **) fun->args, fun->res, fun->int_work, fun->float_work, NULL);
    return 0;
}




int ocp_problem_acados_solve(ocp_problem_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int ocp_problem_acados_setup_qp_matrices_and_factorize(ocp_problem_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}




void ocp_problem_acados_batch_solve(ocp_problem_solver_capsule ** capsules, int * status_out, int N_batch, int num_threads_in_batch_solve)
{
    int num_threads_bkp;
    if (num_threads_in_batch_solve > 1)
    {
        num_threads_bkp = omp_get_num_threads();
        omp_set_num_threads(num_threads_in_batch_solve);
    }

    #pragma omp parallel for
    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_solve(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }

    if (num_threads_in_batch_solve > 1)
    {
        omp_set_num_threads( num_threads_bkp );
    }
    return;
}


void ocp_problem_acados_batch_setup_qp_matrices_and_factorize(ocp_problem_solver_capsule ** capsules, int * status_out, int N_batch, int num_threads_in_batch_solve)
{
    int num_threads_bkp;
    if (num_threads_in_batch_solve > 1)
    {
        num_threads_bkp = omp_get_num_threads();
        omp_set_num_threads(num_threads_in_batch_solve);
    }

    #pragma omp parallel for
    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_setup_qp_matrices_and_factorize(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }

    if (num_threads_in_batch_solve > 1)
    {
        omp_set_num_threads( num_threads_bkp );
    }
    return;
}


void ocp_problem_acados_batch_eval_params_jac(ocp_problem_solver_capsule ** capsules, int N_batch, int num_threads_in_batch_solve)
{
    int num_threads_bkp;
    if (num_threads_in_batch_solve > 1)
    {
        num_threads_bkp = omp_get_num_threads();
        omp_set_num_threads(num_threads_in_batch_solve);
    }

    #pragma omp parallel for
    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_params_jac(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }

    if (num_threads_in_batch_solve > 1)
    {
        omp_set_num_threads( num_threads_bkp );
    }
    return;
}



void ocp_problem_acados_batch_eval_solution_sens_adj_p(ocp_problem_solver_capsule ** capsules, const char *field, int stage, double *out, int offset, int N_batch, int num_threads_in_batch_solve)
{
    int num_threads_bkp;
    if (num_threads_in_batch_solve > 1)
    {
        num_threads_bkp = omp_get_num_threads();
        omp_set_num_threads(num_threads_in_batch_solve);
    }

    #pragma omp parallel for
    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_solution_sens_adj_p(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->sens_out, field, stage, out + i*offset);
    }

    if (num_threads_in_batch_solve > 1)
    {
        omp_set_num_threads( num_threads_bkp );
    }
    return;
}


void ocp_problem_acados_batch_set_flat(ocp_problem_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch, int num_threads_in_batch_solve)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_set_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }

    int num_threads_bkp;
    if (num_threads_in_batch_solve > 1)
    {
        num_threads_bkp = omp_get_num_threads();
        omp_set_num_threads(num_threads_in_batch_solve);
    }

    #pragma omp parallel for
    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_set_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }

    if (num_threads_in_batch_solve > 1)
    {
        omp_set_num_threads( num_threads_bkp );
    }
    return;
}



void ocp_problem_acados_batch_get_flat(ocp_problem_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch, int num_threads_in_batch_solve)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_get_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }
    int num_threads_bkp;
    if (num_threads_in_batch_solve > 1)
    {
        num_threads_bkp = omp_get_num_threads();
        omp_set_num_threads(num_threads_in_batch_solve);
    }

    #pragma omp parallel for
    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_get_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }

    if (num_threads_in_batch_solve > 1)
    {
        omp_set_num_threads( num_threads_bkp );
    }
    return;
}



int ocp_problem_acados_free(ocp_problem_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun[i]);
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt[i]);
        
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_jac_p_hess_xu_p[i]);
        
        
        external_function_external_param_casadi_free(&capsule->discr_dyn_phi_fun_jac_ut_xt_hess[i]);
    }
    free(capsule->discr_dyn_phi_fun);
    free(capsule->discr_dyn_phi_fun_jac_ut_xt);
  
    free(capsule->discr_dyn_phi_jac_p_hess_xu_p);
  
    free(capsule->discr_dyn_phi_fun_jac_ut_xt_hess);

    // cost
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac);
    external_function_external_param_casadi_free(&capsule->ext_cost_0_fun_jac_hess);
    
    external_function_external_param_casadi_free(&capsule->ext_cost_0_hess_xu_p);
    
    
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->ext_cost_fun[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->ext_cost_fun_jac_hess[i]);
        
        external_function_external_param_casadi_free(&capsule->ext_cost_hess_xu_p[i]);
        
        
    }
    free(capsule->ext_cost_fun);
    free(capsule->ext_cost_fun_jac);
    free(capsule->ext_cost_fun_jac_hess);
    free(capsule->ext_cost_hess_xu_p);

    // constraints


    external_function_casadi_free(&capsule->p_global_precompute_fun);

    return 0;
}


void ocp_problem_acados_print_stats(ocp_problem_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    double stat[1200];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\talpha");
    if (stat_n > 8)
        printf("\t\tqp_res_stat\tqp_res_eq\tqp_res_ineq\tqp_res_comp");
    printf("\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j == 5 || j == 6)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}

int ocp_problem_acados_custom_update(ocp_problem_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *ocp_problem_acados_get_nlp_in(ocp_problem_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *ocp_problem_acados_get_nlp_out(ocp_problem_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *ocp_problem_acados_get_sens_out(ocp_problem_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *ocp_problem_acados_get_nlp_solver(ocp_problem_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *ocp_problem_acados_get_nlp_config(ocp_problem_solver_capsule* capsule) { return capsule->nlp_config; }
void *ocp_problem_acados_get_nlp_opts(ocp_problem_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *ocp_problem_acados_get_nlp_dims(ocp_problem_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *ocp_problem_acados_get_nlp_plan(ocp_problem_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
