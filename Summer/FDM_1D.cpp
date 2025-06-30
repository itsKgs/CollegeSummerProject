#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int main(){
    double x_0, x_Nx, t_0, t_Nt;
    double dx = 1, dt = 0.01;

    cout<<"Enter Initial Value of X , x_0 : ";
    cin>>x_0;

    cout<<"Enter Final value of X, x_Nx : ";
    cin>>x_Nx;

    cout<<"Enter Initial Value of t , t_0 : ";
    cin>>t_0;

    cout<<"Enter Final value of t, t_Nt : ";
    cin>>t_Nt;

    // dT/dx = k * d2T/dx2
    //T(x, 0) = 27 C
    //T(x_0, t) = 200 C
    //T(x_N, t) = 500 C
    //0 <= t <= 1
    //0 <= x <= 4

    int Nt;
    Nt = round((t_Nt - t_0) / dt);
    cout<<"Total No. of descrete points of time t : "<<Nt + 1<<endl;

    int Nx ;
    Nx = (x_Nx - x_0)/dx;
    cout<<"Total No. of descrete points of x : "<<Nx + 1<<endl;

    double k = 1.0;
    double F;
    F = (k * dt)/(dx * dx);

    cout<<"F = "<<F<<endl;

    vector<vector<double>> T(Nt + 1, vector<double>(Nx + 1, 27.0));


    for (int n = 0; n <= Nt; n++){
        T[n][0] = 200.0;
        T[n][Nx] = 500.0;
    }

    
    for (int n = 0; n < Nt; n++){
        for(int i = 1 ; i < Nx; i++){
            T[n + 1][i] = F * T[n][i-1] + (1-2*F) * T[n][i] + F * T[n][i+1];
        }
    }


    cout << "Time step results (T[i][j]):\n";
    for (int n = 0; n <= Nt; n++) {
        cout << "t = " << n * dt << "s: ";
        for (int i = 0; i <= Nx; i++) {
            cout << T[n][i] << " ";
        }
        cout << endl;
    }


    return 0;
}