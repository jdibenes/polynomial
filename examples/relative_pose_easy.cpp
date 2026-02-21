//*****************************************************************************
// Toy implementation of the minimal 5-point relative pose solver of:
// Li and Hartley. "Five-Point Motion Estimation Made Easy"
// https://users.cecs.anu.edu.au/~hongdong/new5pt_cameraREady_ver_1.pdf
//*****************************************************************************

#include <iostream>
#include <sturm/sturm.h>
#include <polynomial/polynomial.h>
#include <polynomial/helpers_eigen.h>
#include <polynomial/helpers_geometry.h>

bool solver_relative_pose_easy(float const* p1, float const* p2, int& solutions, float* r01, float* t01);

int main()
{
    // Some synthetic data

    Eigen::Matrix<float, 3, 4> pose
    {
        {  0.935014f, 0.1184840f, -0.3342390f,  0.720449f },
        { -0.138680f, 0.9896420f, -0.0371312f,  1.206050f },
        {  0.326376f, 0.0810701f,  0.9417600f, -2.767010f }
    };

    Eigen::Matrix<float, 3, 7> P1{
        { 1.0f,  2.0f, -3.0f, -1.5f, 4.0f, -5.0f,  1.5f},
        { 2.0f, -1.0f, -2.0f,  1.2f, 3.0f,  4.0f, -6.0f},
        {10.0f, 12.0f, 15.0f,  7.0f, 9.0f, 16.0f, 19.0f}
    };

    Eigen::Matrix<float, 3, 3> R_gt = pose(Eigen::seqN(0, 3), Eigen::seqN(0, 3));
    Eigen::Matrix<float, 3, 1> t_gt = pose.col(3);

    // Transform points

    Eigen::Matrix<float, 3, 7> P2 = (R_gt * P1).colwise() + t_gt;

    Eigen::Matrix<float, 2, 7> p1 = P1.colwise().hnormalized();
    Eigen::Matrix<float, 2, 7> p2 = P2.colwise().hnormalized();

    // Estimate pose

    int solutions = 0;
    Eigen::Matrix<float, 3, 10> r_estimated;
    Eigen::Matrix<float, 3, 10> t_estimated;

    bool ok = solver_relative_pose_easy(p1.data(), p2.data(), solutions, r_estimated.data(), t_estimated.data());
    if (!ok) { return -1; }

    Eigen::Matrix<float, 3, 4> pose_estimated;

    // Show results

    for (int i = 0; i < solutions; ++i)
    {
        pose_estimated(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = matrix_R_rodrigues(r_estimated.col(i));
        pose_estimated.col(3) = t_estimated.col(i);

        std::cout << "Solution (" << i << "): " << std::endl;
        std::cout << pose_estimated << std::endl;
        std::cout << std::endl;
    }

    pose.col(3).normalize(); // t is up to scale

    std::cout << "Ground truth: " << std::endl;
    std::cout << pose << std::endl;
    std::cout << std::endl;

    // Done

    return 0;
}

bool solver_relative_pose_easy(float const* p1, float const* p2, int& solutions, float* r_solution, float* t_solution)
{
    ///////////////////////////////////////////////////////////////////////////
    // 1) Load data (five 2D-2D point correpondences)
    ///////////////////////////////////////////////////////////////////////////

    Eigen::Matrix<float, 2, 5> q1 = matrix_from_buffer<float, 2, 5>(p1);
    Eigen::Matrix<float, 2, 5> q2 = matrix_from_buffer<float, 2, 5>(p2);

    ///////////////////////////////////////////////////////////////////////////
    // 2) Compute E nullspace basis
    ///////////////////////////////////////////////////////////////////////////

    Eigen::Matrix<float, 3, 5> Q1 = q1.colwise().homogeneous();
    Eigen::Matrix<float, 3, 5> Q2 = q2.colwise().homogeneous();

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Q = matrix_E_constraints(Q1, Q2); // 5x9 matrix

    Eigen::Matrix<float, 9, 4> e = Q.fullPivLu().kernel();

    ///////////////////////////////////////////////////////////////////////////
    // 3) Model E as the linear combination of basis with unknown coefficients:
    //    E = (w*e(:,0) + x*e(:,1) + y*e(:,2) + z*e(:,3)).reshaped(3, 3)
    //    Consider w = 1 since E is homogeneous, then we solve for x, y, and z
    //    All polynomial coefficients are considered in grevlex order
    ///////////////////////////////////////////////////////////////////////////

    Eigen::Matrix<x38::polynomial<float, 3>, 3, 3> E = x38::matrix_to_polynomial_grevlex<float, 3, 3, 3>(e);

    ///////////////////////////////////////////////////////////////////////////
    // 4) Apply hidden variable technique for z
    //    Coefficients of E are now polynomials in (x, y) with coefficients 
    //    that are polynomials in (z)
    ///////////////////////////////////////////////////////////////////////////

    int const hidden_variable_index = 2; // 0: x, 1: y, 2: z

    Eigen::Matrix<x38::polynomial<x38::polynomial<float, 1>, 2>, 3, 3> E_hidden = x38::hide_in(E, hidden_variable_index);

    ///////////////////////////////////////////////////////////////////////////
    // 5) Compute E constraints:
    //    det(E) == 0 which is 
    //                a scalar
    //    2*E*E.transpose()*E - (E*E.transpose()).trace() * E == 0 which is a
    //                                                             9x9 matrix
    ///////////////////////////////////////////////////////////////////////////

    x38::polynomial<x38::polynomial<float, 1>, 2> E_determinant = E_hidden.determinant();

    Eigen::Matrix<x38::polynomial<x38::polynomial<float, 1>, 2>, 3, 3> EEt = E_hidden * E_hidden.transpose();
    Eigen::Matrix<x38::polynomial<x38::polynomial<float, 1>, 2>, 3, 3> E_singular_values = (EEt * E_hidden) - ((0.5 * EEt.trace()) * E_hidden);

    ///////////////////////////////////////////////////////////////////////////
    // 6) Stack E constraints into a 10x10 matrix with monomials:
    //    [1, x, y, x^2, x*y, y^2, x^3, x^2*y, x*y^2, y^3].reverse()
    ///////////////////////////////////////////////////////////////////////////

    Eigen::Matrix<x38::polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S(10, 10);

    S << x38::matrix_from_polynomial_grevlex<x38::polynomial<float, 1>, 9, 10>(E_singular_values).rowwise().reverse(),
         x38::matrix_from_polynomial_grevlex<x38::polynomial<float, 1>, 1, 10>(E_determinant).rowwise().reverse();

    std::cout << "E constraints (10x10): " << std::endl;
    std::cout << S << std::endl;
    std::cout << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // 7) Solution requires that det(S(z)) == 0 which yields a univariate
    //    polynomial in z of degree 10
    //    For grevlex order, the matrix S(z) has the following structure:
    //    - First 4 columns are constants
    //    - Next 3 columns are degree 1 polynomials in z
    //    - Next 2 columns are degree 2 polynomials in z
    //    - Last column contains degree 3 polynomials in z
    //    Compute the determinant of the 10x10 matrix S(z) using Gaussian
    //    elimination (determinant is product of diagonal elements)
    ///////////////////////////////////////////////////////////////////////////

    // 7.1) Eliminate first 4 columns

    for (int i = 0; i < 4; ++i) { x38::row_echelon_step(S, i, i, { 0 }, false); }

    Eigen::Matrix<x38::polynomial<float, 1>, Eigen::Dynamic, Eigen::Dynamic> S6 = S(Eigen::seqN(4, 6), Eigen::seqN(4, 6));

    std::cout << "E constraints after 4 Gaussian Elimination steps: " << std::endl;
    std::cout << S << std::endl;
    std::cout << std::endl;

    std::cout << "Extracting bottom right 6x6 matrix: " << std::endl;
    std::cout << S6 << std::endl;
    std::cout << std::endl;

    // 7.2) Eliminate next 3 columns

    x38::row_echelon_step(S6, 0, 0, { 1 }, true);
    x38::row_echelon_step(S6, 1, 0, { 0 }, true);

    x38::row_echelon_step(S6, 2, 1, { 1 }, true);
    x38::row_echelon_step(S6, 3, 1, { 0 }, true);

    x38::row_echelon_step(S6, 4, 2, { 1 }, true);
    x38::row_echelon_step(S6, 5, 2, { 0 }, true);

    std::cout << "E constraints after 3 Gaussian Elimination steps for monomials (z,1): " << std::endl;
    std::cout << S6 << std::endl;
    std::cout << std::endl;

    x38::polynomial<float, 1> hidden_variable = x38::monomial<float, 1>{ 1, { 1 } };

    S6.row(1) = (S6.row(1) * hidden_variable) - S6.row(0);
    S6.row(3) = (S6.row(3) * hidden_variable) - S6.row(2);
    S6.row(5) = (S6.row(5) * hidden_variable) - S6.row(4);

    std::cout << "E constraints after eliminating (1) by multiplying by (z): " << std::endl;
    std::cout << S6 << std::endl;
    std::cout << std::endl;

    S6.row(1).swap(S6.row(2));
    S6.row(2).swap(S6.row(4));

    std::cout << "E constraints after reordering eliminated (1)'s: " << std::endl;
    std::cout << S6 << std::endl;
    std::cout << std::endl;

    Eigen::Matrix<x38::polynomial<float, 1>, 3, 3> S3 = S6(Eigen::seqN(3, 3), Eigen::seqN(3, 3));

    std::cout << "Extracting bottom right 3x3 matrix: " << std::endl;
    std::cout << S3 << std::endl;
    std::cout << std::endl;

    // 7.3) Obtain z polynomial from determinant of remaining 3x3 matrix

    x38::polynomial<float, 1> hidden_univariate = S3.determinant();

    Eigen::Matrix<float, 1, 11> hidden_coefficients = x38::matrix_from_polynomial_grevlex<float, 1, 11>(hidden_univariate);

    std::cout << "Determinant of 3x3 matrix is: " << hidden_univariate << std::endl;
    std::cout << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // 8) Find real roots of z polynomial (degree 10)
    ///////////////////////////////////////////////////////////////////////////    

    double coefficients[11];
    double z_roots[10];
    int n_roots = 0;
    for (int i = 0; i < 11; ++i) { coefficients[i] = hidden_coefficients(i); }
    if (!find_real_roots_sturm(coefficients, 10, z_roots, &n_roots, 2, 0) || (n_roots <= 0))
    {
        std::cout << "No real roots found!" << std::endl;
        return false;
    }

    std::cout << "Found " << n_roots << " real roots: ";
    for (int i = 0; i < n_roots; ++i) { std::cout << z_roots[i] << " "; }
    std::cout << std::endl << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // 9) Compute pose (R,t) from real roots of z polynomial
    ///////////////////////////////////////////////////////////////////////////

    solutions = n_roots;

    Eigen::Matrix<float, 10, 1> monomial_eigenvector;
    Eigen::Matrix<float, 3, 3> E_estimated;

    result_R_t_from_E<float> result;

    Eigen::Matrix<float, 3, 3> R;
    Eigen::Matrix<float, 3, 1> t;
    Eigen::Matrix<float, 3, 1> r;

    for (int i = 0; i < n_roots; ++i)
    {
        float z = static_cast<float>(z_roots[i]);

        // 9.1) Plug (z) into S and get (x,y) from eigenvector for the smallest
        //      singular value

        monomial_eigenvector = x38::slice(x38::substitute(S, { true }, x38::monomial_values<float, 1>{ z }), {}).bdcSvd(Eigen::ComputeFullV).matrixV().col(9);

        float x = monomial_eigenvector(8) / monomial_eigenvector(9);
        float y = monomial_eigenvector(7) / monomial_eigenvector(9);

        // 9.2) Plug (x,y,z) into E

        E_estimated = x38::slice(x38::substitute(E, { true, true, true }, x38::merge(x38::array_type<float, 2>{ x, y }, hidden_variable_index, z)), {});

        // 9.3) Decompose E into (R,t)

        result = R_t_from_E(E_estimated, q1, q2);

        R = result.P(Eigen::all, Eigen::seqN(0, 3));
        t = result.P.col(3);
        r = vector_r_rodrigues(R);

        // 9.4) Store solutions

        matrix_to_buffer(r, r_solution);
        matrix_to_buffer(t, t_solution);

        r_solution += 3;
        t_solution += 3;
    }

    ///////////////////////////////////////////////////////////////////////////
    // 10) Done
    ///////////////////////////////////////////////////////////////////////////

    return true;
}
