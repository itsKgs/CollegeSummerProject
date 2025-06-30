# Dictionary

##construct Dictionary
#:var :- symbol datatype
dog = Dict("name" => :eggdog, :age => 3, 23 => "egg-dog mix")

##cant sort Dictionary
##dictinary is unordered collection
##access value using key

dog["name"] 
dog[:age]
dog[23]

#mute value using key
dog["name"] = "doggo"
dog[:age] = [1,2,3]
dog[23] = :breed

## add new key-value pair
dog[:email] = "doggg@doog.dog"

dog

##remove a key value pair
pop!(dog, 23)
dog

#struct: user defined named fields

## declear struct

mutable struct Dog
    name::String
    age:: Integer
    breed::String
end

# create struct instance
myDog = Dog(
    "eggdog",
    3,
    "egg-dog mix"
)

typeof(myDog)

## acess struct field value
myDog.name
myDog.age
myDog.breed

## mutate struct field value
myDog.name = "Doggo"

myDog

## cant't add new fields to struct

#myDog.email = "doogggo@dog"

#conditions

x,y = 1,2
#x,y = y,x

#x=y 

#tasks
 
task_1() = println("$x > $y")
task_2() = println("$x < $y")
task_3() = println("$x == $y")

# if expression
if x > y
    task_1()
elseif x < y 
    task_2()
else
    task_3()
end

x,y = 1,2
x,y = y,x

x=y

#Ternary operators
x,y = 1,2
x,y = y,x

task_1() = println("$x > $y")
task_2() = println("$x < $y")
task_3() = println("$x == $y")


x > y ? task_1() : (x < y ? task_2() : task_3())

