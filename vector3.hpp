#pragma once
template<typename T>
struct Vector3{
    T x, y, z;
    Vector3() : x(T(0)), y(T(0)), z(T(0)) {}
    Vector3(const T _x, const T _y, const T _z) : x(_x), y(_y), z(_z) {}
    Vector3(const Vector3 & src) : x(src.x), y(src.y), z(src.z) {}
    Vector3(const T s) : x(s), y(s), z(s) {}
    const Vector3 & operator = (const Vector3 & rhs){
        x = rhs.x;
        y = rhs.y;
        z = rhs.z;
        return (*this);
    }
    Vector3 operator + (const Vector3 & rhs) const{
        return Vector3(x + rhs.x, y + rhs.y, z + rhs.z);
    }
    const Vector3 & operator += (const Vector3 & rhs){
        (*this) = (*this) + rhs;
        return (*this);
    }
    Vector3 operator - (const Vector3 & rhs) const{
        return Vector3(x - rhs.x, y - rhs.y, z - rhs.z);
    }
    const Vector3 & operator -= (const Vector3 & rhs){
        (*this) = (*this) - rhs;
        return (*this);
    }

    Vector3 operator * (const T s) const{
        return Vector3(x * s, y * s, z * s);
    }
    const Vector3 & operator *= (const T s){
        (*this) = (*this) * s;
        return (*this);
    }
    friend Vector3 operator * (const T s, const Vector3 & v){
        return (v * s);
    }
    Vector3 operator / (const T s) const{
        return Vector3(x / s, y / s, z / s);
    }
    const Vector3 & operator /= (const T s){
        (*this) = (*this) / s;
        return (*this);
    }
    
    const Vector3 & operator + () const {
        return (* this);
    }
    const Vector3 operator - () const {
        return Vector3(-x, -y, -z);
    }
    T operator * (const Vector3 & rhs) const{
        return (x * rhs.x) + (y * rhs.y) + (z * rhs.z);
    }
    Vector3 operator ^ (const Vector3 & rhs) const{
        return Vector3( (y * rhs.z - z * rhs.y),
                        (z * rhs.x - x * rhs.z),
                        (x * rhs.y - y * rhs.x) );
    }
    const T & operator[](const int i) const {
        return (&x)[i];
    }
    T & operator[](const int i){
        return (&x)[i];
    }
    
    friend std::ostream & operator <<(std::ostream & c, const Vector3 & u){
        c<<u.x<<"   "<<u.y<<"    "<<u.z;
        return c;
    }
    friend std::istream & operator >>(std::istream & c, Vector3 & u){
        c>>u.x; c>>u.y; c>>u.z;
        return c;
    }
};

using F64vec = Vector3<double>;
using F32vec = Vector3<float>;
