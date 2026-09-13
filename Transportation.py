import numpy as np


# ============================================================
# VOGEL'S APPROXIMATION METHOD (VAM)
# ============================================================

def vogel_approximation(cost, supply, demand):

    cost = np.array(cost, dtype=float)

    supply = np.array(supply, dtype=float).copy()
    demand = np.array(demand, dtype=float).copy()

    m, n = cost.shape

    allocation = np.zeros((m, n))

    active_rows = set(range(m))
    active_cols = set(range(n))


    while active_rows and active_cols:

        candidates = []


        # ----------------------------------------------------
        # Calculate row penalties
        # ----------------------------------------------------

        for i in active_rows:

            values = sorted(
                cost[i, j] for j in active_cols
            )

            if len(values) >= 2:
                penalty = values[1] - values[0]
            else:
                penalty = values[0]

            candidates.append(
                ("row", i, penalty)
            )


        # ----------------------------------------------------
        # Calculate column penalties
        # ----------------------------------------------------

        for j in active_cols:

            values = sorted(
                cost[i, j] for i in active_rows
            )

            if len(values) >= 2:
                penalty = values[1] - values[0]
            else:
                penalty = values[0]

            candidates.append(
                ("column", j, penalty)
            )


        # ----------------------------------------------------
        # Select maximum penalty
        # ----------------------------------------------------

        def selection_key(item):

            typ, index, penalty = item

            if typ == "row":

                cheapest = min(
                    cost[index, j]
                    for j in active_cols
                )

            else:

                cheapest = min(
                    cost[i, index]
                    for i in active_rows
                )

            return (penalty, -cheapest)


        typ, index, penalty = max(
            candidates,
            key=selection_key
        )


        # ----------------------------------------------------
        # Select minimum cost cell
        # ----------------------------------------------------

        if typ == "row":

            i = index

            j = min(
                active_cols,
                key=lambda j: cost[i, j]
            )

        else:

            j = index

            i = min(
                active_rows,
                key=lambda i: cost[i, j]
            )


        # ----------------------------------------------------
        # Allocate
        # ----------------------------------------------------

        quantity = min(
            supply[i],
            demand[j]
        )

        allocation[i, j] = quantity

        supply[i] -= quantity
        demand[j] -= quantity


        # Remove satisfied row/column
        if abs(supply[i]) < 1e-9:

            active_rows.remove(i)

        if abs(demand[j]) < 1e-9:

            active_cols.remove(j)


    return allocation


# ============================================================
# FIND CLOSED LOOP FOR MODI
# ============================================================

def find_cycle(basic, start, m, n):

    allowed = set(basic)

    allowed.add(start)


    def dfs(path, move_row):

        i, j = path[-1]


        if move_row:

            candidates = [
                (i, j2)
                for j2 in range(n)
                if (i, j2) in allowed
            ]

        else:

            candidates = [
                (i2, j)
                for i2 in range(m)
                if (i2, j) in allowed
            ]


        for cell in candidates:

            # We found a complete cycle
            if cell == start and len(path) >= 4:

                return path + [start]


            if cell in path:

                continue


            result = dfs(
                path + [cell],
                not move_row
            )


            if result:

                return result


        return None


    return dfs([start], True)


# ============================================================
# HANDLE DEGENERACY
# ============================================================

def add_degenerate_basic(allocation, m, n):

    basic = {

        (i, j)

        for i in range(m)

        for j in range(n)

        if allocation[i, j] > 1e-9
    }


    for i in range(m):

        for j in range(n):

            if len(basic) >= m + n - 1:

                break


            if (i, j) in basic:

                continue


            # Add zero allocation if it does not create a cycle
            if find_cycle(
                basic,
                (i, j),
                m,
                n
            ) is None:

                basic.add((i, j))

                allocation[i, j] = 0


        if len(basic) >= m + n - 1:

            break


    return basic


# ============================================================
# MODI METHOD
# ============================================================

def modi_method(cost, allocation):

    cost = np.array(cost, dtype=float)

    allocation = np.array(
        allocation,
        dtype=float
    )


    m, n = cost.shape


    # --------------------------------------------------------
    # Basic variables
    # --------------------------------------------------------

    basic = add_degenerate_basic(
        allocation,
        m,
        n
    )


    while True:

        # ----------------------------------------------------
        # Find u and v
        # ----------------------------------------------------

        u = [None] * m
        v = [None] * n

        u[0] = 0

        changed = True


        while changed:

            changed = False

            for i, j in basic:

                if u[i] is not None and v[j] is None:

                    v[j] = cost[i, j] - u[i]

                    changed = True


                elif v[j] is not None and u[i] is None:

                    u[i] = cost[i, j] - v[j]

                    changed = True


        # ----------------------------------------------------
        # Calculate opportunity costs
        # Δij = cij - ui - vj
        # ----------------------------------------------------

        delta = np.full(
            (m, n),
            np.nan
        )


        for i in range(m):

            for j in range(n):

                if (i, j) not in basic:

                    delta[i, j] = (
                        cost[i, j]
                        - u[i]
                        - v[j]
                    )


        print("\n--------------------------------")
        print("MODI Iteration")
        print("--------------------------------")

        print("u values:", u)
        print("v values:", v)

        print("\nOpportunity Cost Matrix:")
        print(delta)


        # ----------------------------------------------------
        # Check optimality
        # ----------------------------------------------------

        minimum_delta = np.nanmin(delta)


        if minimum_delta >= -1e-9:

            print("\nOptimal solution reached.")

            return (
                allocation,
                np.sum(allocation * cost),
                u,
                v,
                delta
            )


        # ----------------------------------------------------
        # Entering variable
        # Most negative delta
        # ----------------------------------------------------

        entering = np.unravel_index(
            np.nanargmin(delta),
            delta.shape
        )


        print(
            "\nEntering cell:",
            entering
        )


        # ----------------------------------------------------
        # Find closed loop
        # ----------------------------------------------------

        cycle = find_cycle(
            basic,
            entering,
            m,
            n
        )


        if cycle is None:

            raise RuntimeError(
                "Unable to find a closed loop."
            )


        print("Closed loop:", cycle)


        # ----------------------------------------------------
        # + and - positions
        # ----------------------------------------------------

        plus_cells = cycle[0:-1:2]

        minus_cells = cycle[1:-1:2]


        # ----------------------------------------------------
        # Find theta
        # ----------------------------------------------------

        theta = min(
            allocation[i, j]
            for i, j in minus_cells
        )


        print("Theta =", theta)


        # ----------------------------------------------------
        # Update allocation
        # ----------------------------------------------------

        for k, (i, j) in enumerate(
            cycle[:-1]
        ):

            if k % 2 == 0:

                allocation[i, j] += theta

            else:

                allocation[i, j] -= theta


        # ----------------------------------------------------
        # Update basic variables
        # ----------------------------------------------------

        basic.add(entering)


        # Remove a zero-valued basic variable
        for cell in list(basic):

            if (
                cell != entering
                and allocation[cell] <= 1e-9
            ):

                basic.remove(cell)

                break


# ============================================================
# MAIN PROGRAM
# ============================================================

# Transportation cost matrix

cost = np.array([

    [8, 6, 10, 9],

    [9, 12, 13, 7],

    [14, 9, 16, 5]

])


# Supply

supply = [20, 30, 25]


# Demand

demand = [10, 25, 20, 20]


# ============================================================
# CHECK BALANCE
# ============================================================

if sum(supply) != sum(demand):

    print("Transportation problem is unbalanced.")

    exit()


print("======================================")
print("     TRANSPORTATION PROBLEM")
print("======================================")


print("\nCost Matrix:")
print(cost)


print("\nSupply:")
print(supply)


print("\nDemand:")
print(demand)


# ============================================================
# STEP 1: VAM
# ============================================================

initial_solution = vogel_approximation(
    cost,
    supply,
    demand
)


print("\n======================================")
print(" INITIAL BASIC FEASIBLE SOLUTION")
print("======================================")


print(initial_solution)


initial_cost = np.sum(
    initial_solution * cost
)


print(
    "\nInitial transportation cost =",
    initial_cost
)


# ============================================================
# STEP 2: MODI
# ============================================================

optimal_solution, minimum_cost, u, v, delta = modi_method(
    cost,
    initial_solution
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n======================================")
print("       FINAL OPTIMAL SOLUTION")
print("======================================")


print("\nOptimal Shipment Plan:")

print(optimal_solution)


print(
    "\nMinimum Transportation Cost =",
    minimum_cost
)