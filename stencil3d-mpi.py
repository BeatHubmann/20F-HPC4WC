# ******************************************************
#     Program: stencil3d
#     Authors: Oliver Fuhrer, Beat Hubmann, Shruti Nath
#        Date: July/August 2020
# Description: Simple stencil example on a cubed sphere
# ******************************************************

import time
import math
import numpy as np
import click
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpi4py import MPI
from partitioner import Partitioner
from cubedspherepartitioner import CubedSpherePartitioner


def laplacian( in_field, lap_field, num_halo, extend=0 ):
    """Compute Laplacian using 2nd-order centered differences.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    num_halo  -- number of halo points
    
    Keyword arguments:
    extend    -- extend computation into halo-zone by this number of points (either 0 or 1).
                 this also uses non-corner halo data to correct stencil for missing corner halo data.
    """

    assert -1 < extend < 2, 'Can only extend Laplacian calculation up to 1 point into halo'

    ib = num_halo - extend
    ie = - num_halo + extend
    jb = num_halo - extend
    je = - num_halo + extend
    
    lap_field[:, jb:je, ib:ie] = - 4. * in_field[:, jb    :je    , ib    :ie]  \
                                      + in_field[:, jb    :je    , ib - 1:ie - 1] \
                                      + in_field[:, jb    :je    , ib + 1:ie + 1 if ie != -1 else None]  \
                                      + in_field[:, jb - 1:je - 1, ib:ie] \
                                      + in_field[:, jb + 1:je + 1 if je != -1 else None, ib:ie] 

    # fixing corners
    if extend > 0:
        # bottom-left
        lap_field[:, jb, num_halo] += in_field[:, num_halo, ib] # lap_field(-1, 0) += in_field(0, -1)
        lap_field[:, num_halo, ib] += in_field[:, jb, num_halo] # lap_field(0, -1) += in_field(-1, 0)
        # bottom-right
        lap_field[:, jb, -num_halo] += in_field[:, num_halo, ie] 
        lap_field[:, num_halo, ie] += in_field[:, jb, -num_halo] 
        # top-left
        lap_field[:, je, num_halo] += in_field[:, -num_halo, ib] 
        lap_field[:, -num_halo, ib] += in_field[:, je, num_halo] 
        # top-right
        lap_field[:, je, -num_halo] += in_field[:, -num_halo, ie] 
        lap_field[:, -num_halo, ie] += in_field[:, je, -num_halo] 



def update_halo( field, num_halo, p=None ):
    """Update the halo-zone. 
    
    field    -- input/output field (nz x ny x nx with halo in x- and y-direction)
    num_halo -- number of halo points
    
    Note: corners are still TO BE DEALT WITH
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # print("Updating halo - I am rank {} on tile {} - RANKS left: {} right: {} up: {} down {}".format(rank, p.tile(), p.left(), p.right(), p.top(), p.bottom()))

    reqs_recv, reqs_send = [], []

    # allocate recv buffers and pre-post the receives (top and bottom edge, without corners)
    b_rcvbuf = np.empty_like(field[:, 0:num_halo, num_halo:-num_halo])
    reqs_recv.append(p.comm().Irecv(b_rcvbuf, source = p.bottom()))
    t_rcvbuf = np.empty_like(field[:, -num_halo:, num_halo:-num_halo])
    reqs_recv.append(p.comm().Irecv(t_rcvbuf, source = p.top()))
    # allocate recv buffers and pre-post the receives (left and right edge, without corners)
    l_rcvbuf = np.empty_like(field[:, num_halo:-num_halo:, 0:num_halo])
    reqs_recv.append(p.comm().Irecv(l_rcvbuf, source = p.left()))
    r_rcvbuf = np.empty_like(field[:, num_halo:-num_halo, -num_halo:])
    reqs_recv.append(p.comm().Irecv(r_rcvbuf, source = p.right()))

    # rotate to fit receiver's halo orientation, then
    # pack and send (top and bottom edge, without corners)
    b_sndbuf = np.rot90(field[:, num_halo:2 * num_halo, num_halo:-num_halo],
                        p.rot_halo_bottom(),
                        axes=(1,2)).copy()
    reqs_send.append(p.comm().Isend(b_sndbuf, dest = p.bottom()))
    t_sndbuf = np.rot90(field[:, -2 * num_halo:-num_halo, num_halo:-num_halo],
                        p.rot_halo_top(),
                        axes=(1,2)).copy()
    reqs_send.append(p.comm().Isend(t_sndbuf, dest = p.top()))

    # rotate to fit receiver's halo orientation, then
    # pack and send (left and right edge, without corners)
    l_sndbuf = np.rot90(field[:, num_halo:-num_halo, num_halo:2 * num_halo],
                        p.rot_halo_left(),
                        axes=(1,2)).copy()
   # if p.global_rank() == 4:
   #      with np.printoptions(precision=5, suppress=True, linewidth=120):
   #          # print("Left buffer data:\n{}".format(field[0, num_halo:-num_halo, num_halo:2 * num_halo])) 
   #          print("l_sndbuf:\n{}".format(l_sndbuf[0]))
    reqs_send.append(p.comm().Isend(l_sndbuf, dest = p.left()))    
    r_sndbuf = np.rot90(field[:, num_halo:-num_halo, -2 * num_halo:-num_halo],
                        p.rot_halo_right(),
                        axes=(1,2)).copy()
    reqs_send.append(p.comm().Isend(r_sndbuf, dest = p.right()))
       
    # wait and unpack
    for req in reqs_recv:
        req.wait()

    
    field[:,         0: num_halo, num_halo:-num_halo] = b_rcvbuf
    field[:, -num_halo:         , num_halo:-num_halo] = t_rcvbuf
    field[:,  num_halo:-num_halo,        0: num_halo] = l_rcvbuf
    field[:,  num_halo:-num_halo,-num_halo:         ] = r_rcvbuf

  #  if p.global_rank() == 1:
  #       with np.printoptions(precision=5, suppress=True, linewidth=120):
  #           # print("Left buffer data:\n{}".format(field[0, num_halo:-num_halo, num_halo:2 * num_halo])) 
  #           print("r_rcvbuf:\n{}".format(r_rcvbuf[0]))
  #           print("inserted:\n{}".format( field[0,  num_halo:-num_halo,-num_halo:]))
    # wait for sends to complete
    for req in reqs_send:
        req.wait()
            

def apply_diffusion( in_field, out_field, alpha, num_halo, num_iter=1, p=None ):
    """Integrate 4th-order diffusion equation by a certain number of iterations.
    
    in_field  -- input field (nz x ny x nx with halo in x- and y-direction)
    lap_field -- result (must be same size as in_field)
    alpha     -- diffusion coefficient (dimensionless)
    
    Keyword arguments:
    num_iter  -- number of iterations to execute
    p -- responsible partitioner instance 
    """

    tmp_field = np.zeros_like( in_field )
    
    for n in range(num_iter):
        
        update_halo( in_field, num_halo, p )
        
        laplacian( in_field, tmp_field, num_halo=num_halo, extend=1 )
        # update_halo( tmp_field, num_halo, p )
        laplacian( tmp_field, out_field, num_halo=num_halo, extend=0 )

        # DEBUG:
        if p.global_rank() == 0 and n % 10 == 0:
            with np.printoptions(precision=3, suppress=True, linewidth=120):
                # print(np.flipud(in_field[0]))
                print('Max in_field = {} at {}'.format(np.amax(in_field[0]), np.where(in_field[0] == np.amax(in_field[0]))))
            # print('Max tmp_field = {} at {}'.format(np.amax(tmp_field[0]), np.where(tmp_field[0] == np.amax(tmp_field[0]))))
            # print('Max out_field = {} at {}'.format(np.amax(out_field[0]), np.where(out_field[0] == np.amax(out_field[0]))))


        out_field[:, num_halo:-num_halo, num_halo:-num_halo] = \
            in_field[:, num_halo:-num_halo, num_halo:-num_halo] \
            - alpha * out_field[:, num_halo:-num_halo, num_halo:-num_halo]

        # DEBUG:
        # if p.global_rank() == 0:
        #     if n % 10 == 0:
        #         print('Max post alpha out_field = {} at {}'.format(np.amax(out_field[0]), np.where(out_field[0] == np.amax(out_field[0]))))
        #         print(20*'-')
        
        
        if n < num_iter - 1:
            in_field, out_field = out_field, in_field

            
@click.command()
@click.option('--nx', type=int, required=True, help='Number of gridpoints in x-direction')
@click.option('--ny', type=int, required=True, help='Number of gridpoints in y-direction')
@click.option('--nz', type=int, required=True, help='Number of gridpoints in z-direction')
@click.option('--num_iter', type=int, required=True, help='Number of iterations')
@click.option('--num_halo', type=int, default=2, help='Number of halo-pointers in x- and y-direction')
@click.option('--plot_result', type=bool, default=False, help='Make a plot of the result?')
@click.option('--verify', type=bool, default=False, help='Output verification plots? No diffusion, overrides plot_result option')
def main(nx, ny, nz, num_iter, num_halo=2, plot_result=False, verify=False):
    """Driver for apply_diffusion that sets up fields and does timings"""
    
    assert 0 < nx <= 1024*1024, 'You have to specify a reasonable value for nx'
    assert 0 < ny <= 1024*1024, 'You have to specify a reasonable value for ny'
    assert 0 < nz <= 1024, 'You have to specify a reasonable value for nz'
    assert 0 < num_iter <= 1024*1024, 'You have to specify a reasonable value for num_iter'
    assert 0 < num_halo <= 256, 'Your have to specify a reasonable number of halo points'
    alpha = 1./32.
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    p = CubedSpherePartitioner(comm, [nz, ny, nx], num_halo)

    tile = p.tile()
    local_rank = p.local_rank()

    if local_rank == 0:
        f = np.zeros( (nz, ny + 2 * num_halo, nx + 2 * num_halo) )
        if verify:
            pass
            #test_grid = np.add(*np.mgrid[0:ny * 10:10, 0:nx])
            #f[:, num_halo:-num_halo, num_halo:-num_halo] = test_grid
        else:
            # Option 1: Like stencil2d-mpi during HPC4WC course:
            # f[nz // 4:3 * nz // 4, num_halo + ny // 4:num_halo + 3 * ny // 4, num_halo + nx // 4:num_halo + 3 * nx // 4] = 1.0
            
            # Option 2: Similar to option 1, but positive region extended towards tile edges:
            # f[nz // 10:9 * nz // 10, num_halo + ny // 10:num_halo + 9 * ny // 10, num_halo + nx // 10:num_halo + 9 * nx // 10] = 1.0

            # Option 3: One positive region in bottom-left (0-0) corner, one positive region in top-right (ny-nx) corner         
            f[nz // 4:3 * nz // 4, num_halo:num_halo + ny // 4, num_halo:num_halo + nx // 4] = 1.0
            f[nz // 4:3 * nz // 4, num_halo + 3 * ny // 4:-num_halo, num_halo + 3 * nx // 4:-num_halo] = 1.0

            # Option 4: Similar to option 3, but positive region value is rank- and position-flavored:
            # f[nz // 4:3 * nz // 4, num_halo:num_halo + ny // 4, num_halo:num_halo + nx // 4] = rank * 100 + 125
            # f[nz // 4:3 * nz // 4, num_halo + 3 * ny // 4:-num_halo, num_halo + 3 * nx // 4:-num_halo] = rank * 100 + 175
    else:
        f = np.empty(1)
    in_field = p.scatter(f)

    # to have a rank indication when inspecting fields for verification/diagnostics:
    if verify:
        local_ny, local_nx = in_field.shape[1:]
        test_grid = np.add(*np.mgrid[0:(local_ny - 2 * num_halo) * 100:100, 0:(local_nx - 2 * num_halo)]) + rank / 1e3 
        in_field[:, num_halo:-num_halo, num_halo:-num_halo] = test_grid	

    out_field = np.copy( in_field )

    f = p.gather(in_field)
   
    if local_rank == 0:
      #  if rank == 0:
      #      with np.printoptions(precision=5, suppress=True, linewidth=120):
      #          print("in f:\n{}".format(np.flipud(f[0,:,:])))
        np.save('global_in_field_{}{}'.format(tile, '_verify' if verify else ''), f)
        if plot_result or verify:
            plt.ioff()
            plt.imshow(f[in_field.shape[0] // 2, :, :], origin='lower')
            plt.colorbar()
            plt.savefig('global_in_field_{}{}.png'.format(tile, '_verify' if verify else ''))
            plt.close()

    if plot_result or verify:
        np.save('local_in_field_{}-{}_{}{}'.format(tile, local_rank, rank, '_verify' if verify else ''), in_field)
        plt.ioff()
        plt.imshow(in_field[in_field.shape[0] // 2, :, :], origin='lower')
        plt.colorbar()
        plt.savefig('local_in_field_{}-{}_{}{}'.format(tile, local_rank, rank, '_verify' if verify else ''))
        plt.close()

    # no diffusion performed for verification:
    if not verify:
        # warmup caches
        apply_diffusion( in_field, out_field, alpha, num_halo, p=p )

        comm.Barrier()
        
        # time the actual work
        tic = time.time()
        apply_diffusion( in_field, out_field, alpha, num_halo, num_iter=num_iter, p=p )
        toc = time.time()
        
        comm.Barrier()
        
        if rank == 0:
            print("Elapsed time for work = {} s".format(toc - tic) )

    update_halo(out_field, num_halo, p)

    # halo exchange correctness verification:
    if verify:
        pass


    for r in [0,1,2,3]:
        if p.global_rank() == r:
            with np.printoptions(precision=5, suppress=True, linewidth=120):
                # print("Left buffer data:\n{}".format(field[0, num_halo:-num_halo, num_halo:2 * num_halo])) 
                print("rank {} out_field:\n{}".format(r,np.flipud(out_field[0,:,:])))


    # f = p.gather(out_field)
    # if local_rank == 0:
    #     if rank == 0:
    #         with np.printoptions(precision=5, suppress=True, linewidth=120):
    #             print("out f:\n{}".format(np.flipud(f[0,:,:])))
    #     np.save('out_field_{}{}'.format(tile, '_verify' if verify else ''), f)
    #     if plot_result or verify:
    #         if verify:
    #             plt.imshow(f[out_field.shape[0] // 2, :, :], origin='lower')
    #         else:
    #             plt.imshow(f[out_field.shape[0] // 2, num_halo:-num_halo, num_halo:-num_halo], origin='lower')
    #         plt.colorbar()
    #         plt.savefig('out_field_{}{}.png'.format(tile, '_verify' if verify else ''))
    #         plt.close()

    if plot_result:
        np.save('local_out_field_{}-{}_{}'.format(tile, local_rank, rank), out_field[:, num_halo:-num_halo, num_halo:-num_halo])
        plt.ioff()
        plt.imshow(out_field[out_field.shape[0] // 2, num_halo:-num_halo, num_halo:-num_halo], origin='lower')
        plt.colorbar()
        plt.savefig('local_out_field_{}-{}_{}'.format(tile, local_rank, rank))
        plt.close()


if __name__ == '__main__':
    main()
    


