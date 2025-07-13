#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

int main(){
    double x_0, x_Nx, y_0, y_Ny, t_0, t_Nt;
    double dx = 1, dy = 1, dt = 0.01;

    cout<<"Enter Initial Value of X , x_0 : ";
    cin>>x_0;

    cout<<"Enter Final value of X, x_Nx : ";
    cin>>x_Nx;

    cout<<"Enter Initial Value of Y , y_0 : ";
    cin>>y_0;

    cout<<"Enter Final value of Y, y_Ny : ";
    cin>>y_Ny;

    cout<<"Enter Initial Value of t , t_0 : ";
    cin>>t_0;

    cout<<"Enter Final value of t, t_Nt : ";
    cin>>t_Nt;

    // dT/dx = k * [d2T/dx2 + d2T/dy2]
    //T(x, y, 0) = 27 C
    //T(x_0, y, t) = 200 C
    //T(x, y_0, t) = 200 C
    //T(x_N, y, t) = 500 C
    //T(x, y_0, t) = 500 C
    //0 <= t <= 600
    //0 <= x <= 100

    int Nt;
    Nt = round((t_Nt - t_0) / dt);
    cout<<"Total No. of descrete points of time t : "<<Nt + 1<<endl;

    int Nx;
    Nx = (x_Nx - x_0)/dx;
    cout<<"Total No. of descrete points of x : "<<Nx + 1<<endl;

    int Ny;
    Ny = (y_Ny - y_0)/dy;
    cout<<"Total No. of descrete points of x : "<<Ny + 1<<endl;

    double k = 1.0;
    double F_x;
    F_x = (k * dt)/(dx * dx);

    cout<<"F_x = "<<F_x<<endl;

    double F_y;
    F_y = (k * dt)/(dy * dy);

    cout<<"F_y = "<<F_y<<endl;

    vector<vector<vector<double>>> T(Nt + 1, vector<vector<double>>(Nx + 1, vector<double>(Ny + 1, 27.0)));


    for (int n = 0; n <= Nt; n++){
        for (int i = 0; i <= Nx; i++){
            T[n][i][0] = 200.0;
            T[n][i][Ny] = 500.0;
        }

        for (int j = 0; j <= Ny; j++) {
            T[n][0][j] = 200.0;       
            T[n][Nx][j] = 500.0;      
        }
    }

    
    for(int n = 0; n < Nt; n++) {
        for(int i = 1; i < Nx; i++) {
            for(int j = 1; j < Ny; j++) {
                T[n+1][i][j] = F_x * (T[n][i+1][j] + T[n][i-1][j]) + F_y * (T[n][i][j+1] + T[n][i][j-1]) + (1 - 2*F_x - 2*F_y) * T[n][i][j];
            }
        }
    }


    cout << "\nFinal Temperature Distribution";
    for (int n = 0; n <= Nt; n++){
        cout << "t = " << n * dt << "s: "<<endl;
        for (int i = 0; i <= Nx; i++) {
            for (int j = 0; j <= Ny; j++) {
                cout << T[n][i][j] << "\t";
            }
            cout << endl;
        }
    }
    


    return 0;
}