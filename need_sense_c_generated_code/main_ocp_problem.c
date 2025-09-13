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
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_ocp_problem.h"

// blasfeo
#include "blasfeo_d_aux_ext_dep.h"

#define NX     OCP_PROBLEM_NX
#define NP     OCP_PROBLEM_NP
#define NU     OCP_PROBLEM_NU
#define NBX0   OCP_PROBLEM_NBX0
#define NP_GLOBAL   OCP_PROBLEM_NP_GLOBAL


int main()
{

    ocp_problem_solver_capsule *acados_ocp_capsule = ocp_problem_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = OCP_PROBLEM_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = ocp_problem_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("ocp_problem_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = ocp_problem_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = ocp_problem_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = ocp_problem_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = ocp_problem_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = ocp_problem_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = ocp_problem_acados_get_nlp_opts(acados_ocp_capsule);

    // initial condition
    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 0.9947999715805054;
    ubx0[0] = 0.9947999715805054;
    lbx0[1] = -0.6514000296592712;
    ubx0[1] = -0.6514000296592712;
    lbx0[2] = -0.1761000007390976;
    ubx0[2] = -0.1761000007390976;
    lbx0[3] = 0.6009953618049622;
    ubx0[3] = 0.6009953618049622;
    lbx0[4] = 0.08999930322170258;
    ubx0[4] = 0.08999930322170258;
    lbx0[5] = 0.26459795236587524;
    ubx0[5] = 0.26459795236587524;
    lbx0[6] = 0.7487941980361938;
    ubx0[6] = 0.7487941980361938;
    lbx0[7] = 0.7419000267982483;
    ubx0[7] = 0.7419000267982483;
    lbx0[8] = 0.6315000057220459;
    ubx0[8] = 0.6315000057220459;
    lbx0[9] = 0.4074999988079071;
    ubx0[9] = 0.4074999988079071;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;
    x_init[4] = 0.0;
    x_init[5] = 0.0;
    x_init[6] = 0.0;
    x_init[7] = 0.0;
    x_init[8] = 0.0;
    x_init[9] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];

    // solve ocp in loop
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, N, "x", x_init);
        status = ocp_problem_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat( NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat( NU, N, utraj, NU );
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("ocp_problem_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("ocp_problem_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);

    ocp_problem_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);


    // evaluate adjoint solution sensitivities wrt p_global
    ocp_nlp_out *sens_out = ocp_problem_acados_get_sens_out(acados_ocp_capsule);
    ocp_nlp_out_set_values_to_zero(nlp_config, nlp_dims, sens_out);
    ocp_nlp_eval_params_jac(nlp_solver, nlp_in, nlp_out);
    double tmp_p_global[NP_GLOBAL];
    ocp_nlp_eval_solution_sens_adj_p(nlp_solver, nlp_in, sens_out, "p_global", 0, tmp_p_global);
    printf("\nSucessfully evaluated adjoint solution sensitivities wrt p_global.\n");

    // free solver
    status = ocp_problem_acados_free(acados_ocp_capsule);
    if (status) {
        printf("ocp_problem_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = ocp_problem_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("ocp_problem_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
