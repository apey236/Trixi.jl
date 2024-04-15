# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent
function create_cache_parabolic(mesh::ParallelP4estMesh{3},
                                equations_hyperbolic::AbstractEquations,
                                equations_parabolic::AbstractEquationsParabolic, dg::DG,
                                parabolic_scheme,
                                RealT, ::Type{uEltype}) where {uEltype <: Real}
    # Make sure to balance and partition the p4est and create a new ghost layer before creating any
    # containers in case someone has tampered with the p4est after creating the mesh
    balance!(mesh)
    partition!(mesh)
    update_ghost_layer!(mesh)

    elements = init_elements(mesh, equations_parabolic, dg.basis, uEltype)

    elements = init_elements(mesh, equations_hyperbolic, dg.basis, uEltype)
    interfaces = init_interfaces(mesh, equations_hyperbolic, dg.basis, elements)
    boundaries = init_boundaries(mesh, equations_hyperbolic, dg.basis, elements)

    viscous_container = init_viscous_container_3d(nvariables(equations_hyperbolic),
                                                  nnodes(dg.basis), nelements(elements),
                                                  uEltype)

    cache = (; elements, interfaces, boundaries, viscous_container)
    return cache
end

function calc_gradient!(gradients, u_transformed, t,
                        mesh::ParallelP4estMesh{3}, equations_parabolic,
                        boundary_conditions_parabolic, dg::DG,
                        cache, cache_parabolic)
    gradients_x, gradients_y, gradients_z = gradients

    # Start to receive MPI data
    @trixi_timeit timer() "start MPI receive" start_mpi_receive!(cache.mpi_cache)

    # Prolong solution to MPI interfaces
    @trixi_timeit timer() "prolong2mpiinterfaces" begin
        prolong2mpiinterfaces!(cache, u_transformed, mesh,
                               equations_parabolic, dg.surface_integral, dg)
    end

    # Prolong solution to MPI mortars
    @trixi_timeit timer() "prolong2mpimortars" begin
        prolong2mpimortars!(cache, u_transformed, mesh, equations_parabolic,
                            dg.mortar, dg.surface_integral, dg)
    end

    # Start to send MPI data
    @trixi_timeit timer() "start MPI send" begin
        start_mpi_send!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
    end

    # Reset du
    @trixi_timeit timer() "reset gradients" begin
        reset_du!(gradients_x, dg, cache)
        reset_du!(gradients_y, dg, cache)
        reset_du!(gradients_z, dg, cache)
    end

    # Calculate volume integral
    @trixi_timeit timer() "volume integral" begin
        (; derivative_dhat) = dg.basis
        (; contravariant_vectors) = cache.elements

        @threaded for element in eachelement(dg, cache)

            # Calculate gradients with respect to reference coordinates in one element
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                u_node = get_node_vars(u_transformed, equations_parabolic, dg, i, j, k,
                                       element)

                for ii in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_x, derivative_dhat[ii, i],
                                               u_node, equations_parabolic, dg, ii, j,
                                               k, element)
                end

                for jj in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_y, derivative_dhat[jj, j],
                                               u_node, equations_parabolic, dg, i, jj,
                                               k, element)
                end

                for kk in eachnode(dg)
                    multiply_add_to_node_vars!(gradients_z, derivative_dhat[kk, k],
                                               u_node, equations_parabolic, dg, i, j,
                                               kk, element)
                end
            end

            # now that the reference coordinate gradients are computed, transform them node-by-node to physical gradients
            # using the contravariant vectors
            for k in eachnode(dg), j in eachnode(dg), i in eachnode(dg)
                Ja11, Ja12, Ja13 = get_contravariant_vector(1, contravariant_vectors,
                                                            i, j, k, element)
                Ja21, Ja22, Ja23 = get_contravariant_vector(2, contravariant_vectors,
                                                            i, j, k, element)
                Ja31, Ja32, Ja33 = get_contravariant_vector(3, contravariant_vectors,
                                                            i, j, k, element)

                gradients_reference_1 = get_node_vars(gradients_x, equations_parabolic,
                                                      dg,
                                                      i, j, k, element)
                gradients_reference_2 = get_node_vars(gradients_y, equations_parabolic,
                                                      dg,
                                                      i, j, k, element)
                gradients_reference_3 = get_node_vars(gradients_z, equations_parabolic,
                                                      dg,
                                                      i, j, k, element)

                # note that the contravariant vectors are transposed compared with computations of flux
                # divergences in `calc_volume_integral!`. See
                # https://github.com/trixi-framework/Trixi.jl/pull/1490#discussion_r1213345190
                # for a more detailed discussion.
                gradient_x_node = Ja11 * gradients_reference_1 +
                                  Ja21 * gradients_reference_2 +
                                  Ja31 * gradients_reference_3
                gradient_y_node = Ja12 * gradients_reference_1 +
                                  Ja22 * gradients_reference_2 +
                                  Ja32 * gradients_reference_3
                gradient_z_node = Ja13 * gradients_reference_1 +
                                  Ja23 * gradients_reference_2 +
                                  Ja33 * gradients_reference_3

                set_node_vars!(gradients_x, gradient_x_node, equations_parabolic, dg,
                               i, j, k, element)
                set_node_vars!(gradients_y, gradient_y_node, equations_parabolic, dg,
                               i, j, k, element)
                set_node_vars!(gradients_z, gradient_z_node, equations_parabolic, dg,
                               i, j, k, element)
            end
        end
    end

    # Prolong solution to interfaces
    @trixi_timeit timer() "prolong2interfaces" begin
        prolong2interfaces!(cache_parabolic, u_transformed, mesh,
                            equations_parabolic, dg.surface_integral, dg)
    end

    # Calculate interface fluxes for the gradient. This reuses P4est `calc_interface_flux!` along with a
    # specialization for AbstractEquationsParabolic.
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

    # Prolong solution to mortars. These should reuse the hyperbolic version of `prolong2mortars`
    # !!! NOTE: we reuse the hyperbolic cache here, since it contains both `mortars` and `u_threaded`. 
    # !!! should we have a separate mortars/u_threaded in cache_parabolic?
    @trixi_timeit timer() "prolong2mortars" begin
        prolong2mortars!(cache, u_transformed, mesh, equations_parabolic,
                         dg.mortar, dg.surface_integral, dg)
    end

    # Calculate mortar fluxes. These should reuse the hyperbolic version of `calc_mortar_flux`,
    # along with a specialization on `calc_mortar_flux!` and `mortar_fluxes_to_elements!` for 
    # AbstractEquationsParabolic. 
    @trixi_timeit timer() "mortar flux" begin
        calc_mortar_flux!(cache_parabolic.elements.surface_flux_values,
                          mesh, False(), # False() = no nonconservative terms
                          equations_parabolic,
                          dg.mortar, dg.surface_integral, dg, cache)
    end

    # Finish to receive MPI data
    @trixi_timeit timer() "finish MPI receive" begin
        finish_mpi_receive!(cache.mpi_cache, mesh, equations_parabolic, dg, cache)
    end

    # Calculate MPI interface fluxes
    @trixi_timeit timer() "MPI interface flux" begin
        calc_mpi_interface_parabolic_flux!(cache_parabolic.elements.surface_flux_values,
                                           mesh, equations_parabolic,
                                           dg.surface_integral, dg, cache)
    end

    # Calculate MPI mortar fluxes
    @trixi_timeit timer() "MPI mortar flux" begin
        calc_mpi_mortar_flux!(cache_parabolic.elements.surface_flux_values, mesh,
                              have_nonconservative_terms(equations_parabolic),
                              equations_parabolic,
                              dg.mortar, dg.surface_integral, dg, cache)
    end

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
            for l in eachnode(dg), m in eachnode(dg)
                for v in eachvariable(equations_parabolic)
                    for dim in 1:3
                        grad = gradients[dim]
                        # surface at -x
                        normal_direction = get_normal_direction(1,
                                                                contravariant_vectors,
                                                                1, l, m, element)
                        grad[v, 1, l, m, element] = (grad[v, 1, l, m, element] +
                                                     surface_flux_values[v, l, m, 1,
                                                                         element] *
                                                     factor_1 * normal_direction[dim])

                        # surface at +x
                        normal_direction = get_normal_direction(2,
                                                                contravariant_vectors,
                                                                nnodes(dg), l, m,
                                                                element)
                        grad[v, nnodes(dg), l, m, element] = (grad[v, nnodes(dg), l, m,
                                                                   element] +
                                                              surface_flux_values[v, l,
                                                                                  m,
                                                                                  2,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction[dim])

                        # surface at -y
                        normal_direction = get_normal_direction(3,
                                                                contravariant_vectors,
                                                                l, m, 1, element)
                        grad[v, l, 1, m, element] = (grad[v, l, 1, m, element] +
                                                     surface_flux_values[v, l, m, 3,
                                                                         element] *
                                                     factor_1 * normal_direction[dim])

                        # surface at +y
                        normal_direction = get_normal_direction(4,
                                                                contravariant_vectors,
                                                                l, nnodes(dg), m,
                                                                element)
                        grad[v, l, nnodes(dg), m, element] = (grad[v, l, nnodes(dg), m,
                                                                   element] +
                                                              surface_flux_values[v, l,
                                                                                  m,
                                                                                  4,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction[dim])

                        # surface at -z
                        normal_direction = get_normal_direction(5,
                                                                contravariant_vectors,
                                                                l, m, 1, element)
                        grad[v, l, m, 1, element] = (grad[v, l, m, 1, element] +
                                                     surface_flux_values[v, l, m, 5,
                                                                         element] *
                                                     factor_1 * normal_direction[dim])

                        # surface at +z
                        normal_direction = get_normal_direction(6,
                                                                contravariant_vectors,
                                                                l, m, nnodes(dg),
                                                                element)
                        grad[v, l, m, nnodes(dg), element] = (grad[v, l, m, nnodes(dg),
                                                                   element] +
                                                              surface_flux_values[v, l,
                                                                                  m,
                                                                                  6,
                                                                                  element] *
                                                              factor_2 *
                                                              normal_direction[dim])
                    end
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
        apply_jacobian_parabolic!(gradients_z, mesh, equations_parabolic, dg,
                                  cache_parabolic)
    end

    # Finish to send MPI data
    @trixi_timeit timer() "finish MPI send" finish_mpi_send!(cache.mpi_cache)

    return nothing
end

function calc_mpi_interface_parabolic_flux!(surface_flux_values,
                                  mesh::Union{ParallelP4estMesh{3},
                                              ParallelT8codeMesh{3}},
                                  equations::AbstractEquationsParabolic, 
                                  surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Get element and side index information on the local element
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        # Create the local i,j,k indexing on the local element used to pull normal direction information
        i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                     index_range)
        j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                     index_range)
        k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                     index_range)

        i_element = i_element_start
        j_element = j_element_start
        k_element = k_element_start

        # Initiate the node indices to be used in the surface for loop,
        # the surface flux storage must be indexed in alignment with the local element indexing
        local_surface_indices = surface_indices(local_indices)
        i_surface_start, i_surface_step_i, i_surface_step_j = index_to_start_step_3d(local_surface_indices[1],
                                                                                     index_range)
        j_surface_start, j_surface_step_i, j_surface_step_j = index_to_start_step_3d(local_surface_indices[2],
                                                                                     index_range)
        i_surface = i_surface_start
        j_surface = j_surface_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                # Get the normal direction on the local element
                # Contravariant vectors at interfaces in negative coordinate direction
                # are pointing inwards. This is handled by `get_normal_direction`.
                normal_direction = get_normal_direction(local_direction,
                                                        contravariant_vectors,
                                                        i_element, j_element, k_element,
                                                        local_element)

                calc_mpi_interface_parabolic_flux!(surface_flux_values, mesh,
                                         equations,
                                         surface_integral, dg, cache,
                                         interface, normal_direction,
                                         i, j, local_side,
                                         i_surface, j_surface, local_direction,
                                         local_element)

                # Increment local element indices to pull the normal direction
                i_element += i_element_step_i
                j_element += j_element_step_i
                k_element += k_element_step_i
                # Increment the surface node indices along the local element
                i_surface += i_surface_step_i
                j_surface += j_surface_step_i
            end
            # Increment local element indices to pull the normal direction
            i_element += i_element_step_j
            j_element += j_element_step_j
            k_element += k_element_step_j
            # Increment the surface node indices along the local element
            i_surface += i_surface_step_j
            j_surface += j_surface_step_j
        end
    end

    return nothing
end

# Inlined version of the interface flux computation for conservation laws
@inline function calc_mpi_interface_parabolic_flux!(surface_flux_values,
                                          mesh::Union{ParallelP4estMesh{3},
                                                      ParallelT8codeMesh{3}},
                                          equations::AbstractEquationsParabolic,
                                          surface_integral, dg::DG, cache,
                                          interface_index, normal_direction,
                                          interface_i_node_index,
                                          interface_j_node_index, local_side,
                                          surface_i_node_index, surface_j_node_index,
                                          local_direction_index, local_element_index)
    @unpack u = cache.mpi_interfaces
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg,
                                       interface_i_node_index, interface_j_node_index,
                                       interface_index)

    flux_ = 0.5 * (u_ll + u_rr) # we assume that the gradient computations utilize a central flux

    # Note that we don't flip the sign on the secondary flux. This is because for parabolic terms,
    # the normals are not embedded in `flux_` for the parabolic gradient computations.

    for v in eachvariable(equations)
        surface_flux_values[v, surface_i_node_index, surface_j_node_index,
        local_direction_index, local_element_index] = flux_[v]
    end
end


function calc_mpi_mortar_flux!(surface_flux_values,
                               mesh::Union{ParallelP4estMesh{3}, ParallelT8codeMesh{3}},
                               nonconservative_terms, 
                               equations::AbstractEquationsParabolic,
                               mortar_l2::LobattoLegendreMortarL2,
                               surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_threaded, fstar_tmp_threaded = cache
    index_range = eachnode(dg)

    @threaded for mortar in eachmpimortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar = fstar_threaded[Threads.threadid()]
        fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]

        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        for position in 1:4
            i_small = i_small_start
            j_small = j_small_start
            k_small = k_small_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    # Get the normal direction on the small element.
                    normal_direction = get_normal_direction(cache.mpi_mortars, i, j,
                                                            position, mortar)

                    calc_mpi_mortar_flux!(fstar, mesh, nonconservative_terms, equations,
                                          surface_integral, dg, cache,
                                          mortar, position, normal_direction,
                                          i, j)

                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
            end
            i_small += i_small_step_j
            j_small += j_small_step_j
            k_small += k_small_step_j
        end

        # Buffer to interpolate flux values of the large element to before
        # copying in the correct orientation
        u_buffer = cache.u_threaded[Threads.threadid()]

        mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                       mesh, equations, mortar_l2, dg, cache,
                                       mortar, fstar, u_buffer, fstar_tmp)
    end

    return nothing
end

# Inlined version of the mortar flux computation on small elements for conservation laws
@inline function calc_mpi_mortar_flux!(fstar,
                                       mesh::Union{ParallelP4estMesh{3},
                                                   ParallelT8codeMesh{3}},
                                       nonconservative_terms::False, 
                                       equations::AbstractEquationsParabolic,
                                       surface_integral, dg::DG, cache,
                                       mortar_index, position_index, normal_direction,
                                       i_node_index, j_node_index)
    @unpack u = cache.mpi_mortars
    @unpack surface_flux = surface_integral

    u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, i_node_index,
                                       j_node_index, mortar_index)

    # TODO: parabolic; only BR1 at the moment
    flux_ = 0.5 * (u_ll + u_rr)

    # Copy flux to buffer
    set_node_vars!(fstar, flux_, equations, dg, i_node_index, j_node_index,
                   position_index)
end

@inline function mpi_mortar_fluxes_to_elements!(surface_flux_values,
                                                mesh::Union{ParallelP4estMesh{3},
                                                            ParallelT8codeMesh{3}},
                                                equations::AbstractEquationsParabolic,
                                                mortar_l2::LobattoLegendreMortarL2,
                                                dg::DGSEM, cache, mortar, fstar,
                                                u_buffer, fstar_tmp)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    index_range = eachnode(dg)

    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)
    large_surface_indices = surface_indices(large_indices)

    i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_surface_indices[1],
                                                                           index_range)
    j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_surface_indices[2],
                                                                           index_range)

    for (element, position) in zip(local_neighbor_ids[mortar],
                                   local_neighbor_positions[mortar])
        if position == 5 # -> large element
            # Project small fluxes to large element.
            multiply_dimensionwise!(u_buffer,
                                    mortar_l2.reverse_lower, mortar_l2.reverse_lower,
                                    view(fstar, .., 1),
                                    fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_lower,
                                        view(fstar, .., 2),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_lower,
                                        mortar_l2.reverse_upper,
                                        view(fstar, .., 3),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_upper,
                                        view(fstar, .., 4),
                                        fstar_tmp)
            # The flux is calculated in the outward direction of the small elements,
            # so the sign must be switched to get the flux in outward direction
            # of the large element.
            # The contravariant vectors of the large element (and therefore the normal
            # vectors of the large element as well) are four times as large as the
            # contravariant vectors of the small elements. Therefore, the flux needs
            # to be scaled by a factor of 4 to obtain the flux of the large element.
            # u_buffer .*= -4
            # Copy interpolated flux values from buffer to large element face in the
            # correct orientation.
            # Note that the index of the small sides will always run forward but
            # the index of the large side might need to run backwards for flipped sides.
            i_large = i_large_start
            j_large = j_large_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, i_large, j_large, large_direction, element] = u_buffer[v,
                                                                                                      i,
                                                                                                      j]
                    end
                    i_large += i_large_step_i
                    j_large += j_large_step_i
                end
                i_large += i_large_step_j
                j_large += j_large_step_j
            end
        else # position in (1, 2, 3, 4) -> small element
            # Copy solution small to small
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, i, j, small_direction, element] = fstar[v,
                                                                                       i,
                                                                                       j,
                                                                                       position]
                    end
                end
            end
        end
    end

    return nothing
end

function prolong2mpimortars_divergence!(cache, flux_viscous,
                             mesh::Union{ParallelP4estMesh{3}, ParallelT8codeMesh{3}},
                             equations,
                             mortar_l2::LobattoLegendreMortarL2,
                             surface_integral, dg::DGSEM)
    @unpack node_indices = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    flux_viscous_x, flux_viscous_y, flux_viscous_z = flux_viscous

    @threaded for mortar in eachmpimortar(dg, cache)
        local_neighbor_ids = cache.mpi_mortars.local_neighbor_ids[mortar]
        local_neighbor_positions = cache.mpi_mortars.local_neighbor_positions[mortar]

        # Get start value and step size for indices on both sides to get the correct face
        # and orientation
        small_indices = node_indices[1, mortar]
        small_direction_index = indices2direction(small_indices)
        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        large_indices = node_indices[2, mortar]
        large_direction_index = indices2direction(large_indices)
        i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_indices[1],
                                                                               index_range)
        j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_indices[2],
                                                                               index_range)
        k_large_start, k_large_step_i, k_large_step_j = index_to_start_step_3d(large_indices[3],
                                                                               index_range)

        for (element, position) in zip(local_neighbor_ids, local_neighbor_positions)
            if position == 5 # -> large element
                # Buffer to copy solution values of the large element in the correct orientation
                # before interpolating
                u_buffer = cache.u_threaded[Threads.threadid()]
                # temporary buffer for projections
                fstar_tmp = cache.fstar_tmp_threaded[Threads.threadid()]

                i_large = i_large_start
                j_large = j_large_start
                k_large = k_large_start
                for j in eachnode(dg)
                    for i in eachnode(dg)
                        normal_direction = get_normal_direction(large_direction_index,
                                                        contravariant_vectors,
                                                        i_large, j_large, k_large,
                                                        element)
                        for v in eachvariable(equations)
                            flux_viscous = SVector(flux_viscous_x[v, i_large, j_large, k_large,
                                                          element],
                                           flux_viscous_y[v, i_large, j_large, k_large,
                                                          element],
                                           flux_viscous_z[v, i_large, j_large, k_large,
                                                          element])
                            # We prolong the viscous flux dotted with respect the outward normal 
                            # on the small element. We scale by -1/4 here because the normal 
                            # direction on the large element is negative 4x that of the small 
                            # element (these normal directions are "scaled" by the surface Jacobian)
                            u_buffer[v, i, j] = -0.25 * dot(flux_viscous, normal_direction)
                        end

                        i_large += i_large_step_i
                        j_large += j_large_step_i
                        k_large += k_large_step_i
                    end
                    i_large += i_large_step_j
                    j_large += j_large_step_j
                    k_large += k_large_step_j
                end

                # Interpolate large element face data from buffer to small face locations
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 1, :, :,
                                             mortar),
                                        mortar_l2.forward_lower,
                                        mortar_l2.forward_lower,
                                        u_buffer,
                                        fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 2, :, :,
                                             mortar),
                                        mortar_l2.forward_upper,
                                        mortar_l2.forward_lower,
                                        u_buffer,
                                        fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 3, :, :,
                                             mortar),
                                        mortar_l2.forward_lower,
                                        mortar_l2.forward_upper,
                                        u_buffer,
                                        fstar_tmp)
                multiply_dimensionwise!(view(cache.mpi_mortars.u, 2, :, 4, :, :,
                                             mortar),
                                        mortar_l2.forward_upper,
                                        mortar_l2.forward_upper,
                                        u_buffer,
                                        fstar_tmp)
            else # position in (1, 2, 3, 4) -> small element
                # Copy solution data from the small elements
                i_small = i_small_start
                j_small = j_small_start
                k_small = k_small_start
                for j in eachnode(dg)
                    for i in eachnode(dg)
                        normal_direction = get_normal_direction(small_direction_index,
                                                            contravariant_vectors,
                                                            i_small, j_small, k_small,
                                                            element)
                        for v in eachvariable(equations)
                            flux_viscous = SVector(flux_viscous_x[v, i_small, j_small,
                                                              k_small,
                                                              element],
                                               flux_viscous_y[v, i_small, j_small,
                                                              k_small,
                                                              element],
                                               flux_viscous_z[v, i_small, j_small,
                                                              k_small,
                                                              element])

                            cache.mortars.u[1, v, position, i, j, mortar] = dot(flux_viscous,
                                                                            normal_direction)
                        end
                        i_small += i_small_step_i
                        j_small += j_small_step_i
                        k_small += k_small_step_i
                    end
                    i_small += i_small_step_j
                    j_small += j_small_step_j
                    k_small += k_small_step_j
                end
            end
        end
    end

    return nothing
end

# this version is used for divergence computations
function calc_mpi_interface_flux!(surface_flux_values,
                                  mesh::Union{ParallelP4estMesh{3},
                                              ParallelT8codeMesh{3}},
                                  nonconservative_terms,
                                  equations::AbstractEquationsParabolic,
                                  surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, node_indices, local_sides = cache.mpi_interfaces
    @unpack contravariant_vectors = cache.elements
    index_range = eachnode(dg)

    @threaded for interface in eachmpiinterface(dg, cache)
        # Get element and side index information on the local element
        local_element = local_neighbor_ids[interface]
        local_indices = node_indices[interface]
        local_direction = indices2direction(local_indices)
        local_side = local_sides[interface]

        # Create the local i,j,k indexing on the local element used to pull normal direction information
        i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                     index_range)
        j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                     index_range)
        k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                     index_range)

        i_element = i_element_start
        j_element = j_element_start
        k_element = k_element_start

        # Initiate the node indices to be used in the surface for loop,
        # the surface flux storage must be indexed in alignment with the local element indexing
        local_surface_indices = surface_indices(local_indices)
        i_surface_start, i_surface_step_i, i_surface_step_j = index_to_start_step_3d(local_surface_indices[1],
                                                                                     index_range)
        j_surface_start, j_surface_step_i, j_surface_step_j = index_to_start_step_3d(local_surface_indices[2],
                                                                                     index_range)
        i_surface = i_surface_start
        j_surface = j_surface_start

        for j in eachnode(dg)
            for i in eachnode(dg)
                # We prolong the viscous flux dotted with respect the outward normal on the 
                # primary element. We assume a BR-1 type of flux.
                viscous_flux_normal_ll, viscous_flux_normal_rr = get_surface_node_vars(cache.mpi_interfaces.u,
                                                                                       equations,
                                                                                       dg,
                                                                                       i,
                                                                                       j,
                                                                                       interface)

                flux = 0.5 * (viscous_flux_normal_ll + viscous_flux_normal_rr)

                for v in eachvariable(equations)
                    if local_side == 1
                        # check this line i and j might be i_surface and j_surface
                        surface_flux_values[v, i, j, local_direction, local_element] = flux[v]
                    else
                        surface_flux_values[v, i_surface, j_surface, local_direction, local_element] = -flux[v]
                    end
                end

                # Increment local element indices to pull the normal direction
                i_element += i_element_step_i
                j_element += j_element_step_i
                k_element += k_element_step_i
                # Increment the surface node indices along the local element
                i_surface += i_surface_step_i
                j_surface += j_surface_step_i
            end
            # Increment local element indices to pull the normal direction
            i_element += i_element_step_j
            j_element += j_element_step_j
            k_element += k_element_step_j
            # Increment the surface node indices along the local element
            i_surface += i_surface_step_j
            j_surface += j_surface_step_j
        end
    end

    return nothing
end

function calc_mpi_mortar_flux_divergence!(surface_flux_values,
                               mesh::Union{ParallelP4estMesh{3}, ParallelT8codeMesh{3}},
                               equations::AbstractEquationsParabolic,
                               mortar_l2::LobattoLegendreMortarL2,
                               surface_integral, dg::DG, cache)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    @unpack contravariant_vectors = cache.elements
    @unpack fstar_threaded, fstar_tmp_threaded = cache
    index_range = eachnode(dg)

    @threaded for mortar in eachmpimortar(dg, cache)
        # Choose thread-specific pre-allocated container
        fstar = fstar_threaded[Threads.threadid()]
        fstar_tmp = fstar_tmp_threaded[Threads.threadid()]

        # Get index information on the small elements
        small_indices = node_indices[1, mortar]

        i_small_start, i_small_step_i, i_small_step_j = index_to_start_step_3d(small_indices[1],
                                                                               index_range)
        j_small_start, j_small_step_i, j_small_step_j = index_to_start_step_3d(small_indices[2],
                                                                               index_range)
        k_small_start, k_small_step_i, k_small_step_j = index_to_start_step_3d(small_indices[3],
                                                                               index_range)

        for position in 1:4
            i_small = i_small_start
            j_small = j_small_start
            k_small = k_small_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        viscous_flux_normal_ll = cache.mpi_mortars.u[1, v, position, i, j,
                                                                 mortar]
                        viscous_flux_normal_rr = cache.mpi_mortars.u[2, v, position, i, j,
                                                                 mortar]

                        # TODO: parabolic; only BR1 at the moment
                        fstar[v, i, j, position] = 0.5 * (viscous_flux_normal_ll +
                                                    viscous_flux_normal_rr)
                    end

                    i_small += i_small_step_i
                    j_small += j_small_step_i
                    k_small += k_small_step_i
                end
            end
            i_small += i_small_step_j
            j_small += j_small_step_j
            k_small += k_small_step_j
        end

        # Buffer to interpolate flux values of the large element to before
        # copying in the correct orientation
        u_buffer = cache.u_threaded[Threads.threadid()]

        mpi_mortar_fluxes_to_elements_divergence!(surface_flux_values,
                                       mesh, equations, mortar_l2, dg, cache,
                                       mortar, fstar, u_buffer, fstar_tmp)
    end

    return nothing
end

# # Inlined version of the mortar flux computation on small elements for conservation laws
# @inline function calc_mpi_mortar_flux!(fstar,
#                                        mesh::Union{ParallelP4estMesh{3},
#                                                    ParallelT8codeMesh{3}},
#                                        nonconservative_terms::False, equations,
#                                        surface_integral, dg::DG, cache,
#                                        mortar_index, position_index, normal_direction,
#                                        i_node_index, j_node_index)
#     @unpack u = cache.mpi_mortars
#     @unpack surface_flux = surface_integral

#     u_ll, u_rr = get_surface_node_vars(u, equations, dg, position_index, i_node_index,
#                                        j_node_index, mortar_index)

#     flux = surface_flux(u_ll, u_rr, normal_direction, equations)

#     # Copy flux to buffer
#     set_node_vars!(fstar, flux, equations, dg, i_node_index, j_node_index,
#                    position_index)
# end

@inline function mpi_mortar_fluxes_to_elements_divergence!(surface_flux_values,
                                                mesh::Union{ParallelP4estMesh{3},
                                                            ParallelT8codeMesh{3}},
                                                equations::AbstractEquationsParabolic,
                                                mortar_l2::LobattoLegendreMortarL2,
                                                dg::DGSEM, cache, mortar, fstar,
                                                u_buffer, fstar_tmp)
    @unpack local_neighbor_ids, local_neighbor_positions, node_indices = cache.mpi_mortars
    index_range = eachnode(dg)

    small_indices = node_indices[1, mortar]
    small_direction = indices2direction(small_indices)
    large_indices = node_indices[2, mortar]
    large_direction = indices2direction(large_indices)
    large_surface_indices = surface_indices(large_indices)

    i_large_start, i_large_step_i, i_large_step_j = index_to_start_step_3d(large_surface_indices[1],
                                                                           index_range)
    j_large_start, j_large_step_i, j_large_step_j = index_to_start_step_3d(large_surface_indices[2],
                                                                           index_range)

    for (element, position) in zip(local_neighbor_ids[mortar],
                                   local_neighbor_positions[mortar])
        if position == 5 # -> large element
            # Project small fluxes to large element.
            multiply_dimensionwise!(u_buffer,
                                    mortar_l2.reverse_lower, mortar_l2.reverse_lower,
                                    view(fstar, .., 1),
                                    fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_lower,
                                        view(fstar, .., 2),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_lower,
                                        mortar_l2.reverse_upper,
                                        view(fstar, .., 3),
                                        fstar_tmp)
            add_multiply_dimensionwise!(u_buffer,
                                        mortar_l2.reverse_upper,
                                        mortar_l2.reverse_upper,
                                        view(fstar, .., 4),
                                        fstar_tmp)
            # The flux is calculated in the outward direction of the small elements,
            # so the sign must be switched to get the flux in outward direction
            # of the large element.
            # The contravariant vectors of the large element (and therefore the normal
            # vectors of the large element as well) are four times as large as the
            # contravariant vectors of the small elements. Therefore, the flux needs
            # to be scaled by a factor of 4 to obtain the flux of the large element.
            u_buffer .*= -4
            # Copy interpolated flux values from buffer to large element face in the
            # correct orientation.
            # Note that the index of the small sides will always run forward but
            # the index of the large side might need to run backwards for flipped sides.
            i_large = i_large_start
            j_large = j_large_start
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, i_large, j_large, large_direction, element] = u_buffer[v,
                                                                                                      i,
                                                                                                      j]
                    end
                    i_large += i_large_step_i
                    j_large += j_large_step_i
                end
                i_large += i_large_step_j
                j_large += j_large_step_j
            end
        else # position in (1, 2, 3, 4) -> small element
            # Copy solution small to small
            for j in eachnode(dg)
                for i in eachnode(dg)
                    for v in eachvariable(equations)
                        surface_flux_values[v, i, j, small_direction, element] = fstar[v,
                                                                                       i,
                                                                                       j,
                                                                                       position]
                    end
                end
            end
        end
    end

    return nothing
end

# This is the version used when calculating the divergence of the viscous fluxes
# We pass the `surface_integral` argument solely for dispatch
#(cache, flux_viscous, mesh,equations_parabolic, dg.surface_integral, dg, cache)
function prolong2mpiinterfaces!(cache_parabolic, flux_viscous,
                                mesh::ParallelP4estMesh{3},
                                equations::AbstractEquationsParabolic,
                                surface_integral, dg::DG, cache)
    (; mpi_interfaces) = cache_parabolic
    (; contravariant_vectors) = cache_parabolic.elements
    index_range = eachnode(dg)
    flux_viscous_x, flux_viscous_y, flux_viscous_z = flux_viscous

    @threaded for interface in eachmpiinterface(dg, cache)
        # Copy solution data from the local element using "delayed indexing" with
        # a start value and a step size to get the correct face and orientation.
        # Note that in the current implementation, the interface will be
        # "aligned at the primary element", i.e., the index of the primary side
        # will always run forwards.
        local_side = mpi_interfaces.local_sides[interface]
        local_element = mpi_interfaces.local_neighbor_ids[interface]
        local_indices = mpi_interfaces.node_indices[interface]
        local_direction = indices2direction(local_indices)

        i_element_start, i_element_step_i, i_element_step_j = index_to_start_step_3d(local_indices[1],
                                                                                     index_range)
        j_element_start, j_element_step_i, j_element_step_j = index_to_start_step_3d(local_indices[2],
                                                                                     index_range)
        k_element_start, k_element_step_i, k_element_step_j = index_to_start_step_3d(local_indices[3],
                                                                                     index_range)

        i_element = i_element_start
        j_element = j_element_start
        k_element = k_element_start
        for j in eachnode(dg)
            for i in eachnode(dg)
                # this is the outward normal direction on the primary element
                normal_direction = get_normal_direction(local_direction,
                                                        contravariant_vectors,
                                                        i_element, j_element, k_element,
                                                        local_element)

                for v in eachvariable(equations)
                    flux_viscous = SVector(flux_viscous_x[v, i_element, j_element,
                                                          k_element,
                                                          local_element],
                                           flux_viscous_y[v, i_element, j_element,
                                                          k_element,
                                                          local_element],
                                           flux_viscous_z[v, i_element, j_element,
                                                          k_element,
                                                          local_element])
                    if local_side == 1
                        mpi_interfaces.u[local_side, v, i, j, interface] = dot(flux_viscous,
                                                                        normal_direction)
                    else
                        mpi_interfaces.u[local_side, v, i, j, interface] = -dot(flux_viscous,
                                                                         normal_direction)
                    end

                end
                i_element += i_element_step_i
                j_element += j_element_step_i
                k_element += k_element_step_i
            end
            i_element += i_element_step_j
            j_element += j_element_step_j
            k_element += k_element_step_j
        end
    end

    return nothing
end


end # @muladd
