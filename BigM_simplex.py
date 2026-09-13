import numpy as np


def big_m_simplex(c, constraints, M=1000000):

    c = np.array(c, dtype=float)
    n = len(c)

    # Names of variables
    names = [f"x{i+1}" for i in range(n)]

    processed = []

    # ---------------------------------------------------------
    # Convert constraints to standard form
    # ---------------------------------------------------------
    for i, (a, b, sign) in enumerate(constraints):

        row = list(map(float, a))

        if sign == "<=":

            # Add slack variable
            row.append(1.0)

            names.append(f"s{i+1}")

            basis_idx = len(names) - 1

        elif sign == ">=":

            # Add surplus variable (-1)
            row.append(-1.0)

            names.append(f"s{i+1}")

            # Add artificial variable (+1)
            row.append(1.0)

            names.append(f"A{i+1}")

            basis_idx = len(names) - 1

        elif sign == "=":

            # Add artificial variable
            row.append(1.0)

            names.append(f"A{i+1}")

            basis_idx = len(names) - 1

        else:

            raise ValueError("Constraint must be <=, >= or =")

        processed.append((row, b, basis_idx))


    # ---------------------------------------------------------
    # Create coefficient matrix
    # ---------------------------------------------------------

    total_variables = len(names)

    A = np.zeros((len(processed), total_variables))
    rhs = np.zeros(len(processed))

    basis = []

    for i, (row, b, basis_idx) in enumerate(processed):

        A[i, :len(row)] = row

        rhs[i] = b

        basis.append(basis_idx)


    # ---------------------------------------------------------
    # Objective function coefficients
    # ---------------------------------------------------------

    C = np.zeros(total_variables)

    C[:n] = c

    # Artificial variables receive -M
    for j, name in enumerate(names):

        if name.startswith("A"):

            C[j] = -M


    # ---------------------------------------------------------
    # Calculate Cj - Zj
    # ---------------------------------------------------------

    def reduced_cost():

        cb = C[basis]

        Zj = cb @ A

        Z = cb @ rhs

        Cj_Zj = C - Zj

        return Cj_Zj, Z


    # ---------------------------------------------------------
    # Simplex iterations
    # ---------------------------------------------------------

    for iteration in range(100):

        Cj_Zj, Z = reduced_cost()

        print("\nIteration:", iteration + 1)

        print("Cj - Zj =", Cj_Zj)

        print("Current Z =", Z)

        # Find entering variable
        entering = int(np.argmax(Cj_Zj))

        # If all Cj-Zj <= 0, optimum reached
        if Cj_Zj[entering] <= 1e-9:
            break

        # -----------------------------------------------------
        # Ratio test
        # -----------------------------------------------------

        ratios = []

        for i in range(len(rhs)):

            if A[i, entering] > 1e-12:

                ratio = rhs[i] / A[i, entering]

                ratios.append((ratio, i))


        if not ratios:

            raise ValueError("The problem is unbounded.")


        # Smallest positive ratio
        _, leaving = min(ratios, key=lambda x: x[0])


        print("Entering variable:", names[entering])
        print("Leaving variable:", names[basis[leaving]])


        # -----------------------------------------------------
        # Pivot operation
        # -----------------------------------------------------

        pivot = A[leaving, entering]

        A[leaving] = A[leaving] / pivot
        rhs[leaving] = rhs[leaving] / pivot


        # Make other entries in entering column zero
        for i in range(len(rhs)):

            if i != leaving:

                factor = A[i, entering]

                A[i] = A[i] - factor * A[leaving]

                rhs[i] = rhs[i] - factor * rhs[leaving]


        # Update basis
        basis[leaving] = entering


    # ---------------------------------------------------------
    # Get final solution
    # ---------------------------------------------------------

    x = np.zeros(total_variables)

    for i, j in enumerate(basis):

        x[j] = rhs[i]


    # ---------------------------------------------------------
    # Check artificial variables
    # ---------------------------------------------------------

    for j, name in enumerate(names):

        if name.startswith("A") and x[j] > 1e-7:

            raise ValueError(
                "Problem is infeasible: artificial variable remains positive."
            )


    # ---------------------------------------------------------
    # Display final answer
    # ---------------------------------------------------------

    print("\n===================================")
    print("         BIG-M FINAL ANSWER")
    print("===================================")

    for i in range(n):

        print(f"{names[i]} = {x[i]:.4f}")

    optimal_Z = c @ x[:n]

    print(f"\nOptimal value of Z = {optimal_Z:.4f}")


    return x[:n], optimal_Z


# =============================================================
# CASE STUDY
# =============================================================

# Objective function:
# Max Z = 3x1 + 5x2

c = [3, 5]


# Constraints:
# x1 <= 4
# 2x2 <= 12
# 3x1 + 2x2 >= 18

constraints = [

    ([1, 0], 4, "<="),

    ([0, 2], 12, "<="),

    ([3, 2], 18, ">=")

]


# Solve using Big-M
solution, optimal_value = big_m_simplex(c, constraints)