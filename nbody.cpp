#include<iostream>
#include<fstream>
#include<iomanip>
#include<cassert>
#include<mpi.h>
#include"vector3.hpp"
#include"ic.hpp"
#include"io.hpp"
#include"calc_force.hpp"
#include"integrator.hpp"

struct Ptcl{
    int    id;
    double mass;
    F64vec pos;
    F64vec vel;
    F64vec acc;
    double pot;
    static inline double eps2;
    void set(const int i, const double m, const F64vec & p, const F64vec & v){
        id   = i;
        mass = m;
        pos  = p;
        vel  = v;
    }
    void clear(){
        acc = 0.0;
        pot = 0.0;
    }
    struct Comm{
        double mass;
        F64vec pos;
        template<typename T>
        void copy(T &p){
            mass = p.mass;
            pos  = p.pos;
        }
    };
};

struct Energy{
    double tot;
    double kin;
    double pot;
    Energy() = default;
    template<typename Tptcl>
    Energy(Tptcl ptcl[], const int n){
        this->set(ptcl, n);
    }
    template<typename Tptcl>
    void set(Tptcl ptcl[], const int n){
        double kin_loc = 0.0;
        double pot_loc = 0.0;
        for(int i=0; i<n; i++){
            kin_loc += 0.5 * ptcl[i].mass * ptcl[i].vel * ptcl[i].vel;
            pot_loc += ptcl[i].mass * ptcl[i].pot;
        }
        MPI_Allreduce(&kin_loc, &kin, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&pot_loc, &pot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        pot *= 0.5;
        tot = pot+kin;
    }
    void dump(std::ostream &fout=std::cout){
        fout<<"tot= "<<tot<<" kin= "<<kin<<" pot= "<<pot<<std::endl;
    }
};

template<typename Tpi, typename Tpj>
void calc_gravity(Tpi pi[], const int ni, Tpj pj[], const int nj){
    double eps2 = Tpi::eps2;
    for(int i=0; i<ni; i++){
        for(int j=0; j<nj; j++){
            F64vec rij    = pi[i].pos - pj[j].pos;
            if(rij.x == 0.0 && rij.y == 0.0 && rij.z == 0.0) continue;
            double r3_inv = rij * rij + eps2;
            double r_inv  = 1.0/sqrt(r3_inv);
            r3_inv     = r_inv * r_inv;
            r_inv     *= pj[j].mass;
            r3_inv    *= r_inv;
            pi[i].acc -= r3_inv * rij;
            pi[i].pot -= r_inv;
        }
    }
}

int get_divieded_n(const int n_glb, const int rank, const int n_proc){
    return  (n_glb%n_proc)  <= rank ? n_glb/n_proc : (n_glb/n_proc) + 1;
}

int main(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    std::cout<<std::setprecision(15);
    int n_ptcl_glb = 4096;
    const double dt = 1.0 / 256;
    const double t_end = 50.0;
    const double dt_snap = 1.0 / 8.0;
    auto force_function = calc_gravity<Ptcl, typename Ptcl::Comm>;
    double t_now = 0.0;
    Ptcl::eps2 = 0.01*0.01;
    const double mass_glb = 1.0;
    int id_snap = 0;
    const int my_rank = []{int tmp; MPI_Comm_rank(MPI_COMM_WORLD, &tmp); return tmp;}();
    const int n_proc  = []{int tmp; MPI_Comm_size(MPI_COMM_WORLD, &tmp); return tmp;}();
    Ptcl *ptcl;
#if defined (COLD_SPHERE)
    int n_ptcl_loc = get_divieded_n(n_ptcl_glb, my_rank, n_proc);
    ParticleSystemGenerator model(n_ptcl_loc, mass_glb);
    model.r_cutoff_ = 1.0;
    model.gen_ptcl(ptcl);
#elif defined(PLUMMER_MODEL)
    int n_ptcl_loc = get_divieded_n(n_ptcl_glb, my_rank, n_proc);
    ParticleSystemGenerator model(n_ptcl_loc, mass_glb);
    model.energy_   = -0.25;
    model.gen_ptcl(ptcl, 0, F64vec(0.0), F64vec(0.0), InitialModel::PLUMMER);
#else
    assert(n_ptcl_glb%2 == 0);
    double r_normal_init, v_init;
    double r_peri = 0.5;
    double r_init = 8.0;
    get_parabolic_params(r_normal_init, v_init, mass_glb*2.0, r_peri, r_init);
    double r_horizontal_init = sqrt(r_init*r_init - r_normal_init*r_normal_init);
    F64vec pos_init(0.5*r_horizontal_init, 0.5*r_normal_init, 0.0);
    F64vec vel_init(-0.5*v_init, 0.0, 0.0);
    Ptcl *ptcl_tmp_0, *ptcl_tmp_1;
    int n_ptcl_loc_tmp = get_divieded_n(n_ptcl_glb/2, my_rank, n_proc);
    ParticleSystemGenerator model(n_ptcl_loc_tmp, mass_glb);
    model.energy_   = -0.25;
    model.gen_ptcl(ptcl_tmp_0, 0,             pos_init,  vel_init, InitialModel::PLUMMER);
    model.gen_ptcl(ptcl_tmp_1, n_ptcl_glb/2, -pos_init, -vel_init, InitialModel::PLUMMER);
    int n_ptcl_loc = append_ptcls(ptcl, ptcl_tmp_0, n_ptcl_loc_tmp, ptcl_tmp_1, n_ptcl_loc_tmp);
#endif
    Energy eng_init, eng_now;
    int n_loop_max = t_end / dt;
    double wtime = 0.0;
    calc_force_mpi<typename Ptcl::Comm>(ptcl, n_ptcl_loc, force_function);
    for(int n_loop = 0; n_loop < n_loop_max; n_loop++){
        eng_now.set(ptcl, n_ptcl_loc);
        if(n_loop == 0){
            eng_init = eng_now;
        }
        if(t_now >= id_snap * dt_snap){
            if(my_rank == 0){
                eng_now.dump();
                std::cout<<"t_now= "<<t_now<<" relative_error= "<<(eng_now.tot-eng_init.tot)/eng_init.tot<<std::endl;
            }
            write_snap_mpi(ptcl, n_ptcl_loc, id_snap);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = MPI_Wtime();
        //update_euler(ptcl, n_ptcl_loc, dt, force_function);
        update_rk2(ptcl, n_ptcl_loc, dt, force_function);
        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = MPI_Wtime();
        wtime += t1-t0;
        t_now += dt;
    }
    if(my_rank == 0){
        std::cout<<"wtime= "<<wtime<<" wtime/n_loop_max= "<<wtime/n_loop_max<<std::endl;
        long long n_op_per_interaction = 30;
        long long n_op = (long long)n_ptcl_glb * (long long)n_ptcl_glb * n_op_per_interaction * 2;
        double speed = n_op / (wtime/n_loop_max);
        std::cout<<"speed= "<< speed*1e-9 <<"GFlops"<<std::endl;
    }
    MPI_Finalize();
}
