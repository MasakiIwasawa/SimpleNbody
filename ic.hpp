#pragma once
#include<random>
#include"vector3.hpp"
enum class InitialModel{
    COLD,
    PLUMMER,
    KING,
};

struct ParticleSystemGenerator{
    int n_loc_;
    int n_glb_;
    double mass_glb_;
    double r_cutoff_;
    double energy_;
    std::default_random_engine rand_eng_;
    std::uniform_real_distribution<double> dist_;
    ParticleSystemGenerator(const int n_loc, const double m, int seed=0) : n_loc_(n_loc), mass_glb_(m), rand_eng_(seed), dist_(0.0, 1.0) {
        MPI_Allreduce(&n_loc_, &n_glb_, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    template<typename Tptcl>
    void correct_coord(Tptcl ptcl[]){
        F64vec cm_pos_loc = 0.0;
        F64vec cm_vel_loc = 0.0;
        double cm_mass_loc = 0.0;
        for(int i = 0; i < n_loc_; i++){
            cm_pos_loc  += ptcl[i].mass * ptcl[i].pos;
            cm_vel_loc  += ptcl[i].mass * ptcl[i].vel;
            cm_mass_loc += ptcl[i].mass;
        }
        F64vec cm_pos_glb = 0.0;
        F64vec cm_vel_glb = 0.0;
        double cm_mass_glb = 0.0;
        MPI_Allreduce((double*)&cm_pos_loc,  (double*)&cm_pos_glb,  3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce((double*)&cm_vel_loc,  (double*)&cm_vel_glb,  3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&cm_mass_loc, &cm_mass_glb, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        cm_pos_glb /= cm_mass_glb;
        cm_vel_glb /= cm_mass_glb;
        for(int i = 0; i < n_loc_; i++){
            ptcl[i].pos -= cm_pos_glb;
            ptcl[i].vel -= cm_vel_glb;
        }
    }
    
    void make_cold_sphere_impl(double mass[], F64vec pos[], F64vec vel[], const int n){
        for(int i=0; i<n; i++){
            mass[i] = mass_glb_ / n_glb_;
            do {
                pos[i].x = 2.0*(dist_(rand_eng_)-0.5) * r_cutoff_;
                pos[i].y = 2.0*(dist_(rand_eng_)-0.5) * r_cutoff_;
                pos[i].z = 2.0*(dist_(rand_eng_)-0.5) * r_cutoff_;
            }while(pos[i] * pos[i] >= r_cutoff_*r_cutoff_);
            vel[i].x = 0.0;
            vel[i].y = 0.0;
            vel[i].z = 0.0;
        }    
    }

    void make_plummer_sphere_impl(double mass[], F64vec pos[], F64vec vel[], const int n){
        static const double PI = atan(1.0) * 4.0;
        const double r_scale = -3.0 * PI * mass_glb_ * mass_glb_ / (64.0 * energy_);
        r_cutoff_ = 22.8 / r_scale;
        for(int i=0; i<n; i++){
            mass[i] = mass_glb_ / n_glb_;
            double r_tmp = 9999.9;
            while(r_tmp > r_cutoff_){ 
                double m_tmp = dist_(rand_eng_);
                r_tmp = 1.0 / sqrt( pow(m_tmp, (-2.0/3.0)) - 1.0);
            }
            double phi = 2.0 * PI * dist_(rand_eng_);
            double cth = 2.0 * (dist_(rand_eng_) - 0.5);
            double sth = sqrt(1.0 - cth*cth);
            pos[i].x = r_tmp * sth * cos(phi);
            pos[i].y = r_tmp * sth * sin(phi);
            pos[i].z = r_tmp * cth;
            while(1){
                const double v_max = 0.1;
                const double v_try = dist_(rand_eng_);
                const double v_crit = v_max * dist_(rand_eng_);
                if(v_crit < v_try * v_try * pow( (1.0 - v_try * v_try), 3.5) ){
                    const double ve = sqrt(2.0) * pow( (r_tmp*r_tmp + 1.0), -0.25);
                    phi = 2.0 * PI * dist_(rand_eng_);
                    cth = 2.0 * (dist_(rand_eng_) - 0.5);
                    sth = sqrt(1.0 - cth*cth);
                    vel[i].x = ve * v_try * sth * cos(phi);
                    vel[i].y = ve * v_try * sth * sin(phi);
                    vel[i].z = ve * v_try * cth;
                    break;
                }
            }
        }
        const double coef = 1.0 / sqrt(r_scale);
        for(int i=0; i<n; i++){
            pos[i] *= r_scale;
            vel[i] *= coef;
        }
    }
    
    template<typename Tptcl>
    void gen_ptcl(Tptcl *&ptcl, const int id_beg = 0, const F64vec pos_offset = 0.0, const F64vec vel_offset = 0.0, InitialModel initial_model=InitialModel::COLD){
        const int my_rank = []{int tmp; MPI_Comm_rank(MPI_COMM_WORLD, &tmp); return tmp;}();
        const int n_proc  = []{int tmp; MPI_Comm_size(MPI_COMM_WORLD, &tmp); return tmp;}();
        std::vector<int> n_loc_ar(n_proc);
        MPI_Allgather(&n_loc_, 1, MPI_INT, &n_loc_ar[0], 1, MPI_INT, MPI_COMM_WORLD);
        ptcl = new Tptcl[n_loc_];
        int id = id_beg;
        for(int r = 0; r < n_proc; r++){
            int n_tmp = n_loc_ar[r];
            std::vector<double> mass(n_tmp);
            std::vector<F64vec> pos(n_tmp);
            std::vector<F64vec> vel(n_tmp);
            if(my_rank == 0){
                if(initial_model==InitialModel::COLD){
                    make_cold_sphere_impl(&mass[0], &pos[0], &vel[0], n_tmp);
                } else if (initial_model==InitialModel::PLUMMER){
                    make_plummer_sphere_impl(&mass[0], &pos[0], &vel[0], n_tmp);
                }
                if(r == 0){
                    for(int i=0; i<n_tmp; i++){
                        ptcl[i].set(id++, mass[i], pos[i], vel[i]);
                    }
                } else {
                    MPI_Send(&id,  1,       MPI_INT,    r, 0, MPI_COMM_WORLD);
                    MPI_Send(&pos[0],  n_tmp*3, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                    MPI_Send(&vel[0],  n_tmp*3, MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                    MPI_Send(&mass[0], n_tmp,   MPI_DOUBLE, r, 0, MPI_COMM_WORLD);
                    id += n_tmp;
                }
            }
            else if (my_rank == r){
                MPI_Status stat;
                int id_offset;
                MPI_Recv(&id_offset, 1, MPI_INT,    0, 0, MPI_COMM_WORLD, &stat);
                MPI_Recv(&pos[0],  n_tmp*3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
                MPI_Recv(&vel[0],  n_tmp*3, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
                MPI_Recv(&mass[0], n_tmp,   MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &stat);
                for(int i=0; i<n_tmp; i++){
                    ptcl[i].id   = id_offset + i;
                    ptcl[i].pos  = pos[i];
                    ptcl[i].vel  = vel[i];
                    ptcl[i].mass = mass[i];
                }
            }
        }
        correct_coord(ptcl);
        for(int i=0; i<n_loc_; i++){
            ptcl[i].pos += pos_offset;
            ptcl[i].vel += vel_offset;
        }
    }
};

void get_parabolic_params(double & r_normal_init, double & v_init, const double mass_tot, const double r_peri, const double r_init=10.0){
    double v_peri = sqrt(2.0*mass_tot/r_peri);
    double h = v_peri*r_peri;
    v_init = sqrt(2.0*mass_tot/r_init);
    r_normal_init = h / v_init;
}

template<typename Tptcl, typename ... Args>
void get_n_total(int & n_total, Tptcl * ptcl, int n, Args & ... args){
    n_total += n;
    if constexpr (sizeof...(args) >= 2){
        get_n_total(n_total, args...);
    }
}
template<typename Tptcl, typename ... Args>
void append_ptcls_impl(Tptcl *ptcl_dst, int & n_offset, Tptcl *ptcl_src, int n, Args & ... args){
    for(int i=0; i<n; i++){
        ptcl_dst[i+n_offset] = ptcl_src[i];
    }
    n_offset += n;
    if constexpr (sizeof...(args) >= 2){
        append_ptcls_impl(ptcl_dst, n_offset, args...);
    }
}

template<typename Tptcl, typename ... Args>
int append_ptcls(Tptcl *& ptcl, Args & ... args){
    int n_total = 0;
    get_n_total(n_total, args...);
    ptcl = new Tptcl[n_total];
    int n_offset = 0;
    append_ptcls_impl(ptcl, n_offset, args...);
    return n_total;
}
