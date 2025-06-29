println("Hello World")
x = 12
typeof(x)

1 + 1
1-1 

4/2
5/2
2*3

2^3

5%3

# order of operation

1+(2 - 3)^4 * 5

#comparision operator

3 < 2

4>3

4==6

# boolean operator

1==1 && 1>1

1==1 || 1>1

# true =1 and false = 0

false + true + true


x = 1
y = 1

z = x+y 

x = 2

y 

z 

z = x+ y 

# how to incre,ment 

j = 1
j = j + 1
j = j + 1 
j += 1 

x = 1 
typeof(x) 

y = 1.2
typeof(y) 

1/3

typeof(1/3)

# // rational
1//3 
typeof(1//3)

1//3 + 1//7

pi

typeof(pi)  

round(pi; digits = 3)

1_000_00

sqrt(4)

4/2 

5/2 

div(5, 2)

#\div<tab>

4 ÷ 2 

5 ÷ 2


# character & strings

typeof('a')

typeof("a")

typeof("Hello World")

println("doggo says \"bork!\".")

println("""doggo says \"bork!\".""")

#display new line
println("1\n2\n3")

#display tabs

println("1\t2\t3")


# how to concatenate
s1 = "Hello, "
s2 = " World!" 

s1_s2 = s1 * s2 

# how to interpolate

s3 = "doggo"

println("$s3 dot jl")

# unicode character

## \alpha<tab>

typeof('α')

## \:dog:<tab>

typeof('🐶')

α = 1
🐶 = 2
α + 🐶 

## \pi<tab>
π 
π == pi

## \euler<tab>

ℯ


# Array

col_vector = [1,2,3]
typeof(col_vector)

row_vector = [4 5 6]
typeof(row_vector)

row_vector = Float16[4.0 5.0 8.0]
typeof(row_vector)

#access vector element 
col_vector[2]

#mutate vector element
col_vector[3] = 20
col_vector

# find the length of vector

length(row_vector)
#sum vector element

sum(col_vector)

# sort vector element
## (descending, nondestructive)

sort(col_vector; rev = true)
col_vector

## descending, destructive)
sort!(col_vector; rev = true)
col_vector

#func_name!()==inplace

# add new element to end of vector
push!(col_vector, 100)

# remove elemet end of the vector
pop!(col_vector)
col_vector

# cobnstruct a matrix

matrix = [1 4 5; 5 6 7]
typeof(matrix)

# acces element in row 1 column 3
matrix[1,3]


#accesss element using column-major
matrix[5]

# construct array with elements of different data types

multitytpes = [1, 1.0, 1//3, π, 'a', "doggo", [7,8,9], [2 3 4]]
typeof(multitytpes)

# Tuple
## construct Tuple
dog = ("eggod", 3, "3ggg-dog mix")
dog[1] 
## tupple elements are immutable

#dog[1] =  10

#sort(dog)

## tuple size is immutable
#push!(1, 23, 5)

# destructure assignment
(name, age, breed) = dog
name 
age
breed

 # named tuple
 ## construct named tuple]
dog = (name = "eggdog", age = 3, breed = "egg-dog mix")  
typeof(dog)

##access name tuple element
dog[1]
dog.name
dog.age
dog.breed

## named tuple is immutable
