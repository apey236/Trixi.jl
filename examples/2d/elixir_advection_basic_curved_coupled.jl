
using OrdinaryDiffEq
using Trixi


###############################################################################
# semidiscretization of the linear advection equation

advectionvelocity = (1.0, 1.0)
equations = LinearScalarAdvectionEquation2D(advectionvelocity)

# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
solver = DGSEM(polydeg=3, surface_flux=flux_lax_friedrichs)

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = ( 0.0,  1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (8, 16)

# Create curved mesh with 16 x 16 elements
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

# A semidiscretization collects data structures and functions for the spatial discretization
semi1 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver, 
                                     boundary_conditions=(
                                       Trixi.BoundaryConditionCoupled(2, 1, (:end, :i), Float64),
                                       Trixi.BoundaryConditionCoupled(2, 1, (1, :i),  Float64),
                                       boundary_condition_periodic, boundary_condition_periodic))

coordinates_min = (0.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0,  1.0) # maximum coordinates (max(x), max(y))

cells_per_dimension = (8, 16)

# Create curved mesh with 16 x 16 elements
mesh = CurvedMesh(cells_per_dimension, coordinates_min, coordinates_max)

semi2 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver,
                                     boundary_conditions=(
                                       Trixi.BoundaryConditionCoupled(1, 1, (:end, :i), Float64),
                                       Trixi.BoundaryConditionCoupled(1, 1, (1, :i),  Float64), 
                                       boundary_condition_periodic, boundary_condition_periodic))

# mesh = CurvedMesh((4, 4), coordinates_min, coordinates_max)
# semi3 = SemidiscretizationHyperbolic(mesh, equations, initial_condition_convergence_test, solver)

semi = SemidiscretizationCoupled((semi1, semi2))

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 1.0));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# # The SaveSolutionCallback allows to save the solution to a file in regular intervals
# save_solution = SaveSolutionCallback(interval=100,
#                                      solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=1.6)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, stepsize_callback, analysis_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()
