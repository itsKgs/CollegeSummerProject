using InteractiveUtils  # only needed when using Weave

struct Dual{T}
    val::T   # value
    der::T  # derivative
end

Base.:+(f::Dual, g::Dual) = Dual(f.val + g.val, f.der + g.der)
Base.:+(f::Dual, α::Number) = Dual(f.val + α, f.der)
Base.:+(α::Number, f::Dual) = f + α


#You can also write:
import Base: +
f::Dual + g::Dual = Dual(f.val + g.val, f.der + g.der)

Base.:-(f::Dual, g::Dual) = Dual(f.val - g.val, f.der - g.der)


# Product Rule
Base.:*(f::Dual, g::Dual) = Dual(f.val*g.val, f.der*g.val + f.val*g.der)
Base.:*(α::Number, f::Dual) = Dual(f.val * α, f.der * α)
Base.:*(f::Dual, α::Number) = α * f

# Quotient Rule
Base.:/(f::Dual, g::Dual) = Dual(f.val/g.val, (f.der*g.val - f.val*g.der)/(g.val^2))
Base.:/(α::Number, f::Dual) = Dual(α/f.val, -α*f.der/f.val^2)
Base.:/(f::Dual, α::Number) = f * inv(α) # Dual(f.val/α, f.der * (1/α))

Base.:^(f::Dual, n::Integer) = Base.power_by_squaring(f, n)  # use repeated squaring for integer powers


fd = Dual(3, 4)
gd = Dual(5, 6)

fd + gd

fd * gd

fd * (gd + gd)

# creating a new data structure
add(a1, a2, b1, b2) = (a1+b1, a2+b2)

add(1, 2, 3, 4)

using BenchmarkTools
a, b, c, d = 1, 2, 3, 4
@btime add($(Ref(a))[], $(Ref(b))[], $(Ref(c))[], $(Ref(d))[])

a = Dual(1, 2)
b = Dual(3, 4)

add(j1, j2) = j1 + j2
add(a, b)
@btime add($(Ref(a))[], $(Ref(b))[])

@code_native add(1, 2, 3, 4)

@code_native add(a, b)