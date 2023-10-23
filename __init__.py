
# Native Approach
y = []
ser_len = 13
for x in range(ser_len):
    if (len(y) <2):
        y.append(x)
    else:
        y.append(y[len(y)-1]+y[len(y)-2])
print(y)
y = []
# Dynamic Programming Approach
def dynamic_function(leng):
    if (len(y) == leng):
        print(y)
        return
    elif (len(y) <2):
        y.append(len(y))
    else:
        y.append(y[len(y)-1]+y[len(y)-2])
    dynamic_function(ser_len)

dynamic_function(ser_len)

# Recursive Approach
def Fib(n):
    if n <= 1:
        return n
    else:
        return (Fib(n - 1) + Fib(n - 2))

for i in range(ser_len):
    print(Fib(i),end = " ")

print()
# Prime number
given_num = 11
is_prime = True;
for i in range(given_num):
    if (i > 1 and given_num % i == 0):
        is_prime = False
        break
print(is_prime)

# Function to swap two characters in a character array
def swap(ch, i, j):
    temp = ch[i]
    ch[i] = ch[j]
    ch[j] = temp

# Recursive function to generate all permutations of a string
def permutations(ch, curr_index=0):

    if curr_index == len(ch) - 1:
        print(''.join(ch))

    for i in range(curr_index, len(ch)):
        swap(ch, curr_index, i)
        permutations(ch, curr_index + 1)
        swap(ch, curr_index, i)

s = 'ABCD'
permutations(list(s))

print('abef'.partition('cd'))

print('list ele : ',list(map(lambda x:x**2,range(8))))

queue = ['Amar','Akbar','Anony','ram','iqbal']
queue.pop(0)
print(queue)

print('abcefd'.replace('cd','12'))
tuple = ('abcd',786,2.23,'johm',70.2)
print(tuple[1:3])

print(list(filter(lambda x: x>5,range(8))))



