template<typename Tptcl>
void dump_ptcl(Tptcl ptcl[], int n, std::ostream & fout){
    for(int i=0; i<n; i++){
        fout<<ptcl[i].id<<" "<<ptcl[i].mass<<" "<<ptcl[i].pos<<" "<<ptcl[i].vel<<std::endl;
    }
}

template<typename Tptcl>
void dump_ptcl_mpi(Tptcl ptcl[], const int n_ptcl_loc, std::ofstream & fout){
    const int my_rank = []{int tmp; MPI_Comm_rank(MPI_COMM_WORLD, &tmp); return tmp;}();
    const int n_proc  = []{int tmp; MPI_Comm_size(MPI_COMM_WORLD, &tmp); return tmp;}();
    std::vector<int> n_ptcl_loc_ar(n_proc);
    MPI_Allgather(&n_ptcl_loc, 1, MPI_INT, &n_ptcl_loc_ar[0], 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Status stat;
    for(int r=0; r<n_proc; r++){
        MPI_Barrier(MPI_COMM_WORLD);
        const int n_ptcl_tmp = n_ptcl_loc_ar[r];
        std::vector<Tptcl> ptcl_tmp(n_ptcl_tmp);
        if(my_rank == 0){
            if( r == my_rank ){
                dump_ptcl(&ptcl[0], n_ptcl_tmp, fout);
            } else {
                MPI_Recv(&ptcl_tmp[0], sizeof(Tptcl)*n_ptcl_tmp, MPI_BYTE, r, 0, MPI_COMM_WORLD, &stat);
                dump_ptcl(&ptcl_tmp[0], n_ptcl_tmp, fout);
            }
        } else if (my_rank == r) {
            MPI_Send(ptcl, sizeof(Tptcl)*n_ptcl_tmp,  MPI_BYTE, 0, 0, MPI_COMM_WORLD);            
        }
    }
}

template<typename Tptcl>
void write_snap_mpi(Tptcl ptcl[], const int n_ptcl_loc, int & id_snap){
    char file_name[1024];
    sprintf(file_name, "result/snap%05d.dat", id_snap++);
    std::ofstream fout(file_name);
    dump_ptcl_mpi(ptcl, n_ptcl_loc, fout);
    fout.close();
}
