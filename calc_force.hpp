#pragma once
template<typename Tptclcomm, typename Tptcl, typename Tfunc>
void calc_force_mpi(Tptcl ptcl[], const int n_ptcl_loc, Tfunc func){
    const int my_rank = []{int tmp; MPI_Comm_rank(MPI_COMM_WORLD, &tmp); return tmp;}();
    const int n_proc  = []{int tmp; MPI_Comm_size(MPI_COMM_WORLD, &tmp); return tmp;}();
    std::vector<int> n_ptcl_loc_ar(n_proc);
    MPI_Allgather(&n_ptcl_loc, 1, MPI_INT, &n_ptcl_loc_ar[0], 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<Tptclcomm> ptcl_send(n_ptcl_loc);
    for(int i=0; i<n_ptcl_loc; i++){
        ptcl[i].clear();
        ptcl_send[i].copy(ptcl[i]);
    }
    MPI_Status stat;
    func(ptcl, n_ptcl_loc, &ptcl_send[0], n_ptcl_loc);
    for(int r=1; r<n_proc; r++){
        int rank_send = (my_rank+n_proc+r) % n_proc;
        int rank_recv = (my_rank+n_proc-r) % n_proc;
        int n_ptcl_recv = n_ptcl_loc_ar[rank_recv];
        std::vector<Tptclcomm> ptcl_recv(n_ptcl_recv);
        MPI_Sendrecv(&ptcl_send[0], sizeof(Tptclcomm)*n_ptcl_loc,  MPI_BYTE, rank_send, 0,
                     &ptcl_recv[0], sizeof(Tptclcomm)*n_ptcl_recv, MPI_BYTE, rank_recv, 0, MPI_COMM_WORLD, &stat);
        for(int i=0; i<n_ptcl_recv; i++){
            ptcl_send[i].mass = ptcl_recv[i].mass;
            ptcl_send[i].pos  = ptcl_recv[i].pos;
        }
        func(ptcl, n_ptcl_loc, &ptcl_recv[0], n_ptcl_recv);
    }
}
