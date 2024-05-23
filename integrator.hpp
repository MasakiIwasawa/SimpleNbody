#pragma once
#include"calc_force.hpp"
template<typename Tptcl, typename Tfunc>
void update_euler(Tptcl ptcl[], const int n, const double dt, Tfunc func){
    calc_force_mpi<typename Tptcl::Comm>(ptcl, n, func);
    for(int i=0; i<n; i++){
        ptcl[i].pos += ptcl[i].vel*dt + 0.5*ptcl[i].acc*dt*dt;
        ptcl[i].vel += ptcl[i].acc*dt;
    }
}

template<typename Tptcl, typename Tfunc>
void update_rk2(Tptcl ptcl[], const int n, const double dt, Tfunc func){
    std::vector<F64vec> pos_old(n);
    std::vector<F64vec> vel_old(n);
    std::vector<F64vec> k1x(n);
    std::vector<F64vec> k1v(n);
    for(int i=0; i<n; i++){
        pos_old[i] =ptcl[i].pos;
        vel_old[i] =ptcl[i].vel;
        k1x[i] = ptcl[i].vel * dt;
        k1v[i] = ptcl[i].acc * dt;
        ptcl[i].pos = pos_old[i] + 0.5*k1x[i];
    }
    calc_force_mpi<typename Tptcl::Comm>(ptcl, n, func);
    for(int i=0; i<n; i++){
        auto k2x = dt*(vel_old[i]+0.5*k1v[i]);
        auto k2v = dt*ptcl[i].acc;
        ptcl[i].pos = pos_old[i] + k2x;
        ptcl[i].vel = vel_old[i] + k2v;
    }
    calc_force_mpi<typename Tptcl::Comm>(ptcl, n, func);
}
