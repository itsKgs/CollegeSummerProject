using DifferentialEquations
using Plots
using Sundials, ParameterizedFunctions

function lorenz(du, u, p, t)
    du[1] = p[1]*(u[2] - u[1])
    du[2] = u[1]*(p[2] - u[3]) - u[2]
    du[3] = u[1]*u[2] - p[3]*u[3]
end

u0 = [1.0,0.0,0.0]
tspan = (0.0, 100.0)

p = (10.0,28.0,8/3)

prob = ODEProblem(lorenz, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol = 1e-8)

plot(sol)

plot(sol,vars=(1,2,3))
scatter!(sol,vars=(1,2,3), denseplot=false)

plot(sol,vars=(0,2,3))

scatter!(sol,vars=(0,2,3), denseplot=false)


# N-Body Problems and Astronomy

function pleiades!(du, u, p, t)
    @inbounds begin
        x = view(u, 1:7)   # x
        y = view(u, 8:14)  # y
        v = view(u, 15:21) # x′
        w = view(u, 22:28) # y′
        du[1:7] .= v
        du[8:14] .= w
        for i in 15:28
            du[i] = zero(u[1])
        end

        for i=1:7,j=1:7
            if i != j
                r = ((x[i]-x[j])^2 + (y[i] - y[j])^2)^(3/2)
                du[14+i] += j*(x[j] - x[i])/r
                du[21+i] += j*(y[j] - y[i])/r
            end
        end  
    end  
end


tspan = (0.0,3.0)

u0 = [
    3.0,  3.0,  -1.0, -3.0,  2.0, -2.0,  2.0,   # x positions
    3.0, -3.0,   2.0,  0.0,  0.0, -4.0,  4.0,   # y positions
    0.0,  0.0,   0.0,  0.0,  0.0,  1.75, -1.5,  # x velocities
    0.0,  0.0,   0.0, -1.25, 1.0,  0.0,  0.0    # y velocities
]

prob = ODEProblem(pleiades!, u0, tspan)

sol = solve(prob, Vern8(), abstol=1e-10, reltol=1e-10)

plot(sol)


tspan = (0.0,200.0)
prob = ODEProblem(pleiades!,[3.0,3.0,-1.0,-3.0,2.0,-2.0,2.0,3.0,-3.0,2.0,0,0,-4.0,4.0,0,0,0,0,0,1.75,-1.5,0,0,0,-1.25,1,0,0],tspan)
sol = solve(prob,Vern8(),abstol=1e-10,reltol=1e-10)
plot(sol,vars=((1:7),(8:14)))

plot(sol,vars=((1:7),(8:14)), tspan=(0.0, 2.0))


# Population Ecology: Lotka-Volterra

function lotka(du,u,p,t)
  du[1] = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = -p[3]*u[2] + p[4]*u[1]*u[2]
end

p = [1.5,1.0,3.0,1.0]
prob = ODEProblem(lotka,[1.0,1.0],(0.0,10.0),p)
sol = solve(prob)
plot(sol)


# Biochemistry: Robertson Equations

function rober(du,u,p,t)
  y₁,y₂,y₃ = u
  k₁,k₂,k₃ = p
  du[1] = -k₁*y₁+k₃*y₂*y₃
  du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
  du[3] =  k₂*y₂^2
end
prob = ODEProblem(rober,[1.0,0.0,0.0],(0.0,1e5),(0.04,3e7,1e4))
sol = solve(prob,Rosenbrock23())
plot(sol)

plot(sol, xscale=:log10, tspan=(1e-6, 1e5), layout=(3,1))

# Chemical Physics: Pollution Models

k1=.35e0
k2=.266e2
k3=.123e5
k4=.86e-3
k5=.82e-3
k6=.15e5
k7=.13e-3
k8=.24e5
k9=.165e5
k10=.9e4
k11=.22e-1
k12=.12e5
k13=.188e1
k14=.163e5
k15=.48e7
k16=.35e-3
k17=.175e-1
k18=.1e9
k19=.444e12
k20=.124e4
k21=.21e1
k22=.578e1
k23=.474e-1
k24=.178e4
k25=.312e1
p = (k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25)
function f(dy,y,p,t)
 k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15,k16,k17,k18,k19,k20,k21,k22,k23,k24,k25 = p
 r1  = k1 *y[1]
 r2  = k2 *y[2]*y[4]
 r3  = k3 *y[5]*y[2]
 r4  = k4 *y[7]
 r5  = k5 *y[7]
 r6  = k6 *y[7]*y[6]
 r7  = k7 *y[9]
 r8  = k8 *y[9]*y[6]
 r9  = k9 *y[11]*y[2]
 r10 = k10*y[11]*y[1]
 r11 = k11*y[13]
 r12 = k12*y[10]*y[2]
 r13 = k13*y[14]
 r14 = k14*y[1]*y[6]
 r15 = k15*y[3]
 r16 = k16*y[4]
 r17 = k17*y[4]
 r18 = k18*y[16]
 r19 = k19*y[16]
 r20 = k20*y[17]*y[6]
 r21 = k21*y[19]
 r22 = k22*y[19]
 r23 = k23*y[1]*y[4]
 r24 = k24*y[19]*y[1]
 r25 = k25*y[20]

 dy[1]  = -r1-r10-r14-r23-r24+
          r2+r3+r9+r11+r12+r22+r25
 dy[2]  = -r2-r3-r9-r12+r1+r21
 dy[3]  = -r15+r1+r17+r19+r22
 dy[4]  = -r2-r16-r17-r23+r15
 dy[5]  = -r3+r4+r4+r6+r7+r13+r20
 dy[6]  = -r6-r8-r14-r20+r3+r18+r18
 dy[7]  = -r4-r5-r6+r13
 dy[8]  = r4+r5+r6+r7
 dy[9]  = -r7-r8
 dy[10] = -r12+r7+r9
 dy[11] = -r9-r10+r8+r11
 dy[12] = r9
 dy[13] = -r11+r10
 dy[14] = -r13+r12
 dy[15] = r14
 dy[16] = -r18-r19+r16
 dy[17] = -r20
 dy[18] = r20
 dy[19] = -r21-r22-r24+r23+r25
 dy[20] = -r25+r24
end

u0 = zeros(20)
u0[2]  = 0.2
u0[4]  = 0.04
u0[7]  = 0.1
u0[8]  = 0.3
u0[9]  = 0.01
u0[17] = 0.007
prob = ODEProblem(f,u0,(0.0,60.0),p)
sol = solve(prob,Rodas5())

plot(sol)

plot(sol, xscale=:log10, tspan=(1e-6, 60), layout=(3,1))