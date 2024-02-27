# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
function create_cache_parabolic(mesh::ParallelP4estMesh{2}, equations_hyperbolic::AbstractEquations, equations_parabolic::AbstractEquationsParabolic, dg::DG, parabolic_scheme,
                      RealT, ::Type{uEltype}) where {uEltype <: Real}
    # Make sure to balance and partition the p4est and create a new ghost layer before creating any
    # containers in case someone has tampered with the p4est after creating the mesh
    balance!(mesh)
    partition!(mesh)
    update_ghost_layer!(mesh)

    elements = init_elements(mesh, equations_parabolic, dg.basis, uEltype)

    mpi_interfaces = init_mpi_interfaces(mesh, equations_parabolic, dg.basis, elements)
    mpi_mortars = init_mpi_mortars(mesh, equations_parabolic, dg.basis, elements)
    mpi_cache = init_mpi_cache(mesh, mpi_interfaces, mpi_mortars,
                               nvariables(equations_parabolic), nnodes(dg), uEltype)

    exchange_normal_directions!(mpi_mortars, mpi_cache, mesh, nnodes(dg))

    interfaces = init_interfaces(mesh, equations_parabolic, dg.basis, elements)
    boundaries = init_boundaries(mesh, equations_parabolic, dg.basis, elements)
    mortars = init_mortars(mesh, equations_parabolic, dg.basis, elements)

    viscous_container = init_viscous_container_2d(nvariables(equations_hyperbolic),
                                                  nnodes(dg.basis), nelements(elements),
                                                  uEltype)

    cache = (; elements, interfaces, mpi_interfaces, boundaries, mortars, mpi_mortars,
             mpi_cache)

    # # Add specialized parts of the cache required to compute the volume integral etc.
    # cache = (; cache...,
    #          create_cache_parabolic(mesh, equations_hyperbolic,equations_parabolic, dg, parabolic_scheme, RealT, uEltype)...)
    cache = (; cache...,
             (; elements, interfaces, boundaries, viscous_container)...)
    cache = (; cache..., create_cache(mesh, equations_parabolic, dg.mortar, uEltype)...)

    return cache
end


function rhs_parabolic!(du, u, t, mesh::ParallelP4estMesh{2},
                        equations_parabolic::AbstractEquationsParabolic,
                        initial_condition, boundary_conditions_parabolic, source_terms,
                        dg::DG, parabolic_scheme, cache, cache_parabolic)
    @unpack viscous_container = cache_parabolic
    @unpack u_transformed, gradients, flux_viscous = viscous_container

    # Convert conservative variables to a form more suitable for viscous flux calculations
    @trixi_timeit timer() "transform variables" begin
        transform_variables!(u_transformed, u, mesh, equations_parabolic,
                             dg, parabolic_scheme, cache, cache_parabolic)
    end
    #Compute the gradients of the transformed variables
    @trixi_timeit timer() "calculate gradient" begin
        calc_gradientmpi!(gradients, u_transformed, t, mesh, equations_parabolic,
                       boundary_conditions_parabolic, dg, cache, cache_parabolic)
    end

    # Compute and store the viscous fluxes
    @trixi_timeit timer() "calculate viscous fluxes" begin
        calc_viscous_fluxes!(flux_viscous, gradients, u_transformed, mesh,
                             equations_parabolic, dg, cache, cache_parabolic)
    end

    # The remainder of this function is essentially a regular rhs! for parabolic
    # equations (i.e., it computes the divergence of the viscous fluxes)
    #
    # OBS! In `calc_viscous_fluxes!`, the viscous flux values at the volume nodes of each element have
    # been computed and stored in `fluxes_viscous`. In the following, we *reuse* (abuse) the
    # `interfaces` and `boundaries` containers in `cache_parabolic` to interpolate and store the
    # *fluxes* at the element surfaces, as opposed to interpolating and storing the *solution* (as it
    # is done in the hyperbolic operator). That is, `interfaces.u`/`boundaries.u` store *viscous flux values*
    # and *not the solution*.  The advantage is that a) we do not need to allocate more storage, b) we
    # do not need to recreate the existing data structure only with a different name, and c) we do not
    # need to interpolate solutions *and* gradients to the surfaces.

    # TODO: parabolic; reconsider current data structure reuse strategy

    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache_parabolic.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(cache_parabolic, flux_viscous, mesh, 
            equations_parabolic, dg.surface_integral, dg, cache)
    end

    # TODO: Add the mortar prolongation

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(cache_parabolic.mpi_cache, mesh, equations_parabolic, dg, cache_parabolic)
    end

    # Reset du
    @trixi_timeit timer() "reset ∂u/∂t" reset_du!(du, dg, cache)

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        calc_volume_integral!(du, flux_viscous, mesh, equations_parabolic, dg, cache)
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache_parabolic, flux_viscous, mesh, equations_parabolic,
                            dg.surface_integral, dg, cache)
    end

    # Calculate interface fluxes
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache_parabolic.elements.surface_flux_values, mesh,
                             equations_parabolic, dg, cache_parabolic)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache_parabolic, flux_viscous, mesh, equations_parabolic,
                            dg.surface_integral, dg, cache)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_divergence!(cache_parabolic, t,
                                       boundary_conditions_parabolic, mesh,
                                       equations_parabolic,
                                       dg.surface_integral, dg)
    end

    # Prolong solution to mortars (specialized for AbstractEquationsParabolic)
    # !!! NOTE: we reuse the hyperbolic cache here since it contains "mortars" and "u_threaded". See https://github.com/trixi-framework/Trixi.jl/issues/1674 for a discussion
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars_divergence!(cache, flux_viscous, mesh, equations_parabolic,
                                    dg.mortar, dg.surface_integral, dg)
    end

    # Calculate mortar fluxes (specialized for AbstractEquationsParabolic)
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux_divergence!(cache_parabolic.elements.surface_flux_values,
                                     mesh, equations_parabolic, dg.mortar,
                                     dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(cache_parabolic.mpi_cache, mesh, equations_parabolic, dg, cache_parabolic)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(cache_parabolic.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations_parabolic), equations_parabolic,
                                 dg.surface_integral, dg, cache_parabolic)
    end

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        calc_surface_integral!(du, u, mesh, equations_parabolic,
                               dg.surface_integral, dg, cache_parabolic)
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache_parabolic)
    end

    return nothing
end

function calc_gradientmpi!(gradients, u_transformed, t,
                        mesh::ParallelP4estMesh{2}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG,
                        cache, cache_parabolic)
    gradients_x, gradients_y = gradients

    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache_parabolic.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(cache_parabolic, u_transformed, mesh, 
            equations_parabolic, dg.surface_integral, dg)
    end

    # TODO: Add the mortar prolongation

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(cache_parabolic.mpi_cache, mesh, equations_parabolic, dg, cache_parabolic)
    end

    # Reset du
    @trixi_timeit timer() "reset gradients" begin
        reset_du!(gradients_x, dg, cache)
        reset_du!(gradients_y, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        (; derivative_dhat) = dg.basis
        (; contravariant_vectors) = cache.elements

        @threaded for element in eachelement(dg, cache)

            # Calculate gradients with respect to reference coordinates in one element
            for j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j,
                                       element)

                for ii in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_x, derivative_dhat[ii, i],
                                               u_node,
                                               equations_parabolic, dg, ii, j, element)
                end

                for jj in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_y, derivative_dhat[jj, j],
                                               u_node,
                                               equations_parabolic, dg, i, jj, element)
                end
            end

            # now that the reference coordinate gradients are computed, transform them node-by-node to physical gradients
            # using the contravariant vectors
            for j in eachnode(dg), i in eachnode(dg)
                Ja11, Ja12 = get_contravariant_vector(1, contravariant_vectors, i, j,
                                                      element)
                Ja21, Ja22 = get_contravariant_vector(2, contravariant_vectors, i, j,
                                                      element)

                gradients_reference_1 = get_node_vars(gradients_x, equations_parabolic,
                                                      dg,
                                                      i, j, element)
                gradients_reference_2 = get_node_vars(gradients_y, equations_parabolic,
                                                      dg,
                                                      i, j, element)

                # note that the contravariant vectors are transposed compared with computations of flux
                # divergences in `calc_volume_integral!`. See
                # https://github.com/trixi-framework/Trixi.jl/pull/1490#discussion_r1213345190
                # for a more detailed discussion.
                gradient_x_node = Ja11 * gradients_reference_1 +
                                  Ja21 * gradients_reference_2
                gradient_y_node = Ja12 * gradients_reference_1 +
                                  Ja22 * gradients_reference_2

                set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg, i,
                               j,
                               element)
                set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg, i,
                               j,
                               element)
            end
        end
    end

    # Prolong solution to interfaces. 
    # This reuses `prolong2interfaces` for the purely hyperbolic case.
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache_parabolic, u_transformed, mesh,
                            equations_parabolic, dg.surface_integral, dg)
    end

    # Calculate interface fluxes for the gradient. 
    # This reuses `calc_interface_flux!` for the purely hyperbolic case.
    @trixi_timeit timer() "interface flux" begin
        calc_interface_flux!(cache_parabolic.elements.surface_flux_values,
                             mesh, False(), # False() = no nonconservative terms
                             equations_parabolic, dg.surface_integral, dg,
                             cache_parabolic)
    end

    # Prolong solution to boundaries
    @trixi_timeit timer() "prolong2boundaries" begin
        prolong2boundaries!(cache_parabolic, u_transformed, mesh,
                            equations_parabolic, dg.surface_integral, dg)
    end

    # Calculate boundary fluxes
    @trixi_timeit timer() "boundary flux" begin
        calc_boundary_flux_gradients!(cache_parabolic, t, boundary_conditions_parabolic,
                                      mesh, equations_parabolic, dg.surface_integral,
                                      dg)
    end

    # Prolong solution to mortars. This resues the hyperbolic version of `prolong2mortars`
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u_transformed, mesh, equations_parabolic,
                         dg.mortar, dg.surface_integral, dg)
    end

    # Calculate mortar fluxes. This reuses the hyperbolic version of `calc_mortar_flux`,
    # along with a specialization on `calc_mortar_flux!(fstar, ...)` and `mortar_fluxes_to_elements!` for 
    # AbstractEquationsParabolic. 
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache_parabolic.elements.surface_flux_values,
                          mesh, False(), # False() = no nonconservative terms
                          equations_parabolic,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(cache_parabolic.mpi_cache, mesh, equations_parabolic, dg, cache_parabolic)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_flux!(cache_parabolic.elements.surface_flux_values, mesh,
                                 have_nonconservative_terms(equations_parabolic), equations_parabolic,
                                 dg.surface_integral, dg, cache_parabolic)
    end
  
    # TODO: add mortars mpi interface flux

    # Calculate surface integrals
    @trixi_timeit timer() "surface integral" begin
        (; boundary_interpolation) = dg.basis
        (; surface_flux_values) = cache_parabolic.elements
        (; contravariant_vectors) = cache.elements

        # Access the factors only once before beginning the loop to increase performance.
        # We also use explicit assignments instead of `+=` to let `@muladd` turn these
        # into FMAs (see comment at the top of the file).
        factor_1 = boundary_interpolation[1, 1]
        factor_2 = boundary_interpolation[nnodes(dg), 2]
        @threaded for element in eachelement(dg, cache)
            for l in eachnode(dg)
                for v in eachvariable(equations_parabolic)

                    # Compute x-component of gradients

                    # surface at -x
                    normal_direction_x, _ = get_normal_direction(1,
                                                                 contravariant_vectors,
                                                                 1, l, element)
                    gradients_x[v, 1, l, element] = (gradients_x[v, 1, l, element] +
                                                     surface_flux_values[v, l, 1,
                                                                         element] *
                                                     factor_1 * normal_direction_x)

                    # surface at +x
                    normal_direction_x, _ = get_normal_direction(2,
                                                                 contravariant_vectors,
                                                                 nnodes(dg), l, element)
                    gradients_x[v, nnodes(dg), l, element] = (gradients_x[v, nnodes(dg),
                                                                          l,
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  2,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_x)

                    # surface at -y
                    normal_direction_x, _ = get_normal_direction(3,
                                                                 contravariant_vectors,
                                                                 l, 1, element)
                    gradients_x[v, l, 1, element] = (gradients_x[v, l, 1, element] +
                                                     surface_flux_values[v, l, 3,
                                                                         element] *
                                                     factor_1 * normal_direction_x)

                    # surface at +y
                    normal_direction_x, _ = get_normal_direction(4,
                                                                 contravariant_vectors,
                                                                 l, nnodes(dg), element)
                    gradients_x[v, l, nnodes(dg), element] = (gradients_x[v, l,
                                                                          nnodes(dg),
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  4,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_x)

                    # Compute y-component of gradients

                    # surface at -x
                    _, normal_direction_y = get_normal_direction(1,
                                                                 contravariant_vectors,
                                                                 1, l, element)
                    gradients_y[v, 1, l, element] = (gradients_y[v, 1, l, element] +
                                                     surface_flux_values[v, l, 1,
                                                                         element] *
                                                     factor_1 * normal_direction_y)

                    # surface at +x
                    _, normal_direction_y = get_normal_direction(2,
                                                                 contravariant_vectors,
                                                                 nnodes(dg), l, element)
                    gradients_y[v, nnodes(dg), l, element] = (gradients_y[v, nnodes(dg),
                                                                          l,
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  2,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_y)

                    # surface at -y
                    _, normal_direction_y = get_normal_direction(3,
                                                                 contravariant_vectors,
                                                                 l, 1, element)
                    gradients_y[v, l, 1, element] = (gradients_y[v, l, 1, element] +
                                                     surface_flux_values[v, l, 3,
                                                                         element] *
                                                     factor_1 * normal_direction_y)

                    # surface at +y
                    _, normal_direction_y = get_normal_direction(4,
                                                                 contravariant_vectors,
                                                                 l, nnodes(dg), element)
                    gradients_y[v, l, nnodes(dg), element] = (gradients_y[v, l,
                                                                          nnodes(dg),
                                                                          element] +
                                                              surface_flux_values[v, l,
                                                                                  4,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction_y)
                end
            end
        end
    end

    # Apply Jacobian from mapping to reference element
    @trixi_timeit timer() "Jacobian" begin
        apply_jacobian_parabolic!(gradients_x, mesh, equations_parabolic, dg,
                                  cache_parabolic)
        apply_jacobian_parabolic!(gradients_y, mesh, equations_parabolic, dg,
                                  cache_parabolic)
    end

    return nothing
end

function calc_mpi_interface_flux!(surface_flux_values,
                                  mesh::Union{ParallelP4estMesh{2},
                                              ParallelT8codeMesh{2}},
                                  nonconservative_terms,
                                  equations::AbstractEquationsParabolic,
                                   surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)
    index_end = last(index_range)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Get element and side index information on the local element
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        # Create the local i,j indexing on the local element used to pull normal direction information
        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start

        # Initiate the node index to be used in the surface for loop,
        # the surface flux storage must be indexed in alignment with the local element indexing
        if :i_backward in local_indices
            surface_node = index_end
            surface_node_step = -1
        else
            surface_node = 1
            surface_node_step = 1
        end

        for node in eachnode(dg)
            # Get the normal direction on the local element
            # Contravariant vectors at interfaces in negative coordinate direction
            # are pointing inwards. This is handled by `get_normal_direction`.
            normal_direction = get_normal_direction(local_direction,
                                                    contravariant_vectors,
                                                    i_element, j_element, local_element)

            calc_mpi_interface_flux!(surface_flux_values, mesh, nonconservative_terms,
                                     equations,
                                     surface_integral, dg, cache,
                                     interface, normal_direction,
                                     node, local_side,
                                     surface_node, local_direction, local_element)

            # Increment local element indices to pull the normal direction
            i_element += i_element_step
            j_element += j_element_step

            # Increment the surface node index along the local element
            surface_node += surface_node_step
        end
    end

    return nothing
end

# This version is used for parabolic gradient computations
@inline function calc_mpi_interface_flux!(surface_flux_values,
                                          mesh::Union{ParallelP4estMesh{2},
                                                      ParallelT8codeMesh{2}},
                                          nonconservative_terms::False, 
                                          equations::AbstractEquationsParabolic,
                                          surface_integral, dg::DG, cache,
                                          interface_index, normal_direction,
                                          interface_node_index, local_side,
                                          surface_node_index, local_direction_index,
                                          local_element_index)
    @unpack u = cache.mpi_interfaces
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, interface_node_index,
                                       interface_index)

    flux_ = 0.5 * (u_ll + u_rr) # we assume that the gradient computations utilize a central flux

    # Note that we don't flip the sign on the secondary flux. This is because for parabolic terms,
    # the normals are not embedded in `flux_` for the parabolic gradient computations.
    for v in eachvariable(equations)
        surface_flux_values[v, surface_node_index, local_direction_index, local_element_index] = flux_[v]
    end
end

# This is the version used when calculating the divergence of the viscous fluxes
# We pass the `surface_integral` argument solely for dispatch
function prolong2mpiinterfaces!(cache_parabolic, flux_viscous,
                             mesh::ParallelP4estMesh{2},
                             equations_parabolic::AbstractEquationsParabolic,
                             surface_integral, dg::DG, cache)
    (; mpi_interfaces) = cache_parabolic
    (; contravariant_vectors) = cache_parabolic.elements
    index_range = eachnode(dg)
    flux_viscous_x, flux_viscous_y = flux_viscous

    @threaded for interface in eachmpiinterface(dg, cache)
        # Copy solution data from the primary element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_side = mpi_interfaces.local_sides[interface]
        local_element = mpi_interfaces.local_neighbor_ids[interface]
        local_indices = mpi_interfaces.node_indices[interface]
        # local_element = local_neighbor_ids[interface]
        # local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        # local_side = mpi_interfaces.local_sides[interface]

        i_element_start, i_element_step = index_to_start_step_2d(local_indices[1],
                                                                 index_range)
        j_element_start, j_element_step = index_to_start_step_2d(local_indices[2],
                                                                 index_range)

        i_element = i_element_start
        j_element = j_element_start
        for i in eachnode(dg)

            # this is the outward normal direction on the primary element
            normal_direction = get_normal_direction(local_direction,
                                                    contravariant_vectors,
                                                    i_element, j_element,
                                                    local_element)

            for v in eachvariable(equations_parabolic)
                # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
                flux_viscous = SVector(flux_viscous_x[v, i_element, j_element,
                                                      local_element],
                                       flux_viscous_y[v, i_element, j_element,
                                                      local_element])

                mpi_interfaces.u[local_side, v, i, interface] = dot(flux_viscous, normal_direction)
            end
            i_element += i_element_step
            j_element += j_element_step
        end

    #     # Copy solution data from the secondary element using "delayed indexing" with
    #     # a start value and a step size to get the correct face and orientation.
    #     secondary_element = interfaces.neighbor_ids[2, interface]
    #     secondary_indices = interfaces.node_indices[2, interface]
    #     secondary_direction = indices2direction(secondary_indices)

    #     i_secondary_start, i_secondary_step = index_to_start_step_2d(secondary_indices[1],
    #                                                                  index_range)
    #     j_secondary_start, j_secondary_step = index_to_start_step_2d(secondary_indices[2],
    #                                                                  index_range)

    #     i_secondary = i_secondary_start
    #     j_secondary = j_secondary_start
    #     for i in eachnode(dg)
    #         # This is the outward normal direction on the secondary element.
    #         # Here, we assume that normal_direction on the secondary element is
    #         # the negative of normal_direction on the primary element.
    #         normal_direction = get_normal_direction(secondary_direction,
    #                                                 contravariant_vectors,
    #                                                 i_secondary, j_secondary,
    #                                                 secondary_element)

    #         for v in eachvariable(equations_parabolic)
    #             # OBS! `interfaces.u` stores the interpolated *fluxes* and *not the solution*!
    #             flux_viscous = SVector(flux_viscous_x[v, i_secondary, j_secondary,
    #                                                   secondary_element],
    #                                    flux_viscous_y[v, i_secondary, j_secondary,
    #                                                   secondary_element])
    #             # store the normal flux with respect to the primary normal direction
    #             interfaces.u[2, v, i, interface] = -dot(flux_viscous, normal_direction)
    #         end
    #         i_secondary += i_secondary_step
    #         j_secondary += j_secondary_step
    #     end
    end

    return nothing
end



end # @muladd