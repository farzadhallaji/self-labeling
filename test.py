# sum([0,2.0])
from itertools import product
product(['+','-', ''], repeat=8)

#########################################

# def sunn(l) :
# 	s = 0
# 	for i in l :
# 		s += i
# 	return s

# print(sunn(([1,2,3])))

# def sunn(l) :
# 	s = 0
# 	while len(l):
# 		s += l.pop()

# 	return s

# print(sunn(([1,2,3])))


# def sunn(l) :
# 	if len(l) == 0:
# 		return 0
# 	return l.pop() + sunn(l)


# print(sunn(([1,2,3])))

#########################################  2

# def merge(l1,l2):
# 	s = []
# 	l1 = iter(l1)
# 	l2 = iter(l2)
# 	for it1,it2 in zip(l1,l2):
# 		s.append(it1)
# 		s.append(it2)
# 	return s

# print(merge(['a', 'b', 'c'] , [3, 2, 1]))

#########################################  3

# def fibo(n):
# 	l = []
# 	a,b = 0,1
# 	while n:
# 		l.append(a)
# 		a,b = b, a+b
# 		n -= 1
# 	return l
# print(fibo(20))

########################################  4
# def mx(l) :
# 	atos = [str(aa) for aa in l]
# 	atos.sort(reverse = True)
# 	return int(''.join(atos))#.sort(key = operator.itemgetter(0)))

# print(mx([50 ,2 ,1 ,9]))

#######################################  5


# Problem 1
# ---------
# Write three functions that compute the sum of the numbers in a given list
# using a for-loop, a while-loop, and recursion.
def problem1_1(l):
    s = 0
    for n in l:
        s += n
    return s


def problem1_2(l):
    s, i = 0, 0
    while i < len(l):
        s += l[i]
        i += 1
    return s


def problem1_3(l, accu=0):
    if len(l):
        it = iter(l)
        accu += it.next()
        return problem1_3(list(it), accu)
    else:
        return accu


def problem1_real_programmer(l):
    return sum(l)


# Problem 2
# ---------
# Write a function that combines two lists by alternatingly taking elements.
# For example: given the two lists [a, b, c] and [1, 2, 3], the function
# should return [a, 1, b, 2, c, 3]
def problem2(l1, l2):
    result = []
    for (e1, e2) in zip(l1, l2):
        result.append(e1)
        result.append(e2)
    return result


def problem2_real_programmer(l1, l2):
    from itertools import chain
    return [x for x in chain.from_iterable(zip(l1, l2))]


# Problem 3
# ---------
# Write a function that computes the list of the first 100 Fibonacci numbers.
# By definition, the first two numbers in the Fibonacci sequence are 0 and 1,
# and each subsequent number is the sum of the previous two. As an example,
# here are the first 10 Fibonnaci numbers: 0, 1, 1, 2, 3, 5, 8, 13, 21, and 34.
def fibonacci():
    a, b = 0, 1
    while 1:
        yield a
        a, b = b, a + b


def problem3():
    fib = fibonacci()
    return [fib.next() for _ in range(100)]


# Problem 4
# ---------
# Write a function that given a list of non negative integers, arranges them
# such that they form the largest possible number. For example, given
# [50, 2, 1, 9], the largest formed number is 95021.
def problem4(l):
    # convert one time to string to avoid multiple casting during comparison
    ls = sorted(map(str, l), cmp=lambda e, f: cmp(e + f, f + e), reverse=True)
    return int(''.join(ls))


# Problem 5
# ---------
# Write a program that outputs all possibilities to put + or - or nothing
# between the numbers 1, 2, ..., 9 (in this order) such that the result is
# always 100. For example: 1 + 2 + 34 – 5 + 67 – 8 + 9 = 100.
def problem5():
    from itertools import product
    results, numbers = [], range(1, 10)
    for perm in product(['+', '-', ''], repeat=8):  # iterate on arrangements of operators
        tuples = zip(numbers, perm + ('',))  # add something for digit 9
        expression = ''.join([str(e1) + e2 for (e1, e2) in tuples])  # create expression as string
        if eval(expression) == 100:  # you know what this does
            results.append(expression + ' = 100')
    return results


# Check the solutions
# -------------------
assert (problem1_1([1, 2, 3, 4, 5, 6]) == sum([1, 2, 3, 4, 5, 6]))
assert (problem1_2([1, 2, 3, 4, 5, 6]) == sum([1, 2, 3, 4, 5, 6]))
assert (problem1_3([1, 2, 3, 4, 5, 6]) == sum([1, 2, 3, 4, 5, 6]))
assert (problem1_real_programmer([1, 2, 3, 4, 5, 6]) == sum([1, 2, 3, 4, 5, 6]))
assert (problem2(['a', 'b', 'c'], [1, 2, 3]) == ['a', 1, 'b', 2, 'c', 3])
assert (problem2_real_programmer(['a', 'b', 'c'], [1, 2, 3]) == ['a', 1, 'b', 2, 'c', 3])
assert (len(problem3()) == 100)
assert (problem3()[50] == 12586269025)
assert (problem4([50, 2, 1, 9]) == 95021)
assert (len(problem5()) == 11)
print
"It works!"


def permutation(lst):
    # If lst is empty then there are no permutations
    if len(lst) == 0:
        return []
        # If there is only one element in lst then, only
    # one permuatation is possible
    if len(lst) == 1:
        return [lst]
        # Find the permutations for lst if there are
        # more than 1 characters
    l = []  # empty list that will store current permutation
    # Iterate the input(lst) and calculate the permutation
    for i in range(len(lst)):
        m = lst[i]
        # Extract lst[i] or m from the list.  remLst is
        # remaining list
        remLst = lst[:i] + lst[i + 1:]
        # Generating all permutations where m is first
        # element
        for p in permutation(remLst):
            l.append([m] + p)
    return l


# Driver program to test above function
data = list('123')
for p in permutation(data):
    print
    p
####################################################################################################
# map
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, items))


def multiply(x):
    return (x * x)


def add(x):
    return (x + x)


funcs = [multiply, add]
for i in range(5):
    value = list(map(lambda x: x(i), funcs))
    print(value)
# Output:
# [0, 0]
# [1, 2]
# [4, 4]
# [9, 6]
# [16, 8]

############################
# filter
number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero)

# Output: [-5, -4, -3, -2, -1]

#################################
# reduce
from functools import reduce

product = reduce((lambda x, y: x * y), [1, 2, 3, 4])


# Output: 24

#########################################

# >>> output = None
# >>> msg = output or "No data returned"
# >>> print(msg)
# No data returned

########################################

def a_new_decorator(a_func):
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")

        a_func()

        print("I am doing some boring work after executing a_func()")

    return wrapTheFunction


def a_function_requiring_decoration():
    print("I am the function which needs some decoration to remove my foul smell")


a_function_requiring_decoration()
# outputs: "I am the function which needs some decoration to remove my foul smell"

a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)
# now a_function_requiring_decoration is wrapped by wrapTheFunction()

a_function_requiring_decoration()


# outputs:I am doing some boring work before executing a_func()
#        I am the function which needs some decoration to remove my foul smell
#        I am doing some boring work after executing a_func()

##################
@a_new_decorator
def a_function_requiring_decoration():
    """Hey you! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")


a_function_requiring_decoration()
# outputs: I am doing some boring work before executing a_func()
#         I am the function which needs some decoration to remove my foul smell
#         I am doing some boring work after executing a_func()

# the @a_new_decorator is just a short way of saying:
a_function_requiring_decoration = a_new_decorator(a_function_requiring_decoration)

print(a_function_requiring_decoration.__name__)
# Output: wrapTheFunction

###########
from functools import wraps


def a_new_decorator(a_func):
    @wraps(a_func)
    def wrapTheFunction():
        print("I am doing some boring work before executing a_func()")
        a_func()
        print("I am doing some boring work after executing a_func()")

    return wrapTheFunction


@a_new_decorator
def a_function_requiring_decoration():
    """Hey yo! Decorate me!"""
    print("I am the function which needs some decoration to "
          "remove my foul smell")


print(a_function_requiring_decoration.__name__)
# Output: a_function_requiring_decoration

##################
from functools import wraps


def logit(logfile='out.log'):
    def logging_decorator(func):
        @wraps(func)
        def wrapped_function(*args, **kwargs):
            log_string = func.__name__ + " was called"
            print(log_string)
            # Open the logfile and append
            with open(logfile, 'a') as opened_file:
                # Now we log to the specified logfile
                opened_file.write(log_string + '\n')

        return wrapped_function

    return logging_decorator


@logit()
def myfunc1():
    pass


myfunc1()


# Output: myfunc1 was called
# A file called out.log now exists, with the above string

@logit(logfile='func2.log')
def myfunc2():
    pass


myfunc2()


# Output: myfunc2 was called
# A file called func2.log now exists, with the above string

#########################

class logit(object):
    _logfile = 'out.log'

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        log_string = self.func.__name__ + " was called"
        print(log_string)
        # Open the logfile and append
        with open(self._logfile, 'a') as opened_file:
            # Now we log to the specified logfile
            opened_file.write(log_string + '\n')
        # Now, send a notification
        self.notify()

        # return base func
        return self.func(*args)

    def notify(self):
        # logit only logs, no more
        pass


logit._logfile = 'out2.log'  # if change log file


@logit
def myfunc1():
    pass


myfunc1()
# Output: myfunc1 was called

#######################################################################################

from collections import namedtuple


def profile():
    Person = namedtuple('Person', 'name age')
    return Person(name="Danny", age=31)


# Use as namedtuple
p = profile()
print(p, type(p))
# Person(name='Danny', age=31) <class '__main__.Person'>
print(p.name)
# Danny
print(p.age)
# 31

# Use as plain tuple
p = profile()
print(p[0])
# Danny
print(p[1])
# 31

# Unpack it immediatly
name, age = profile()
print(name)
# Danny
print(age)
# 31

########################################
my_list = [1, 2, 3]
dir(my_list)
# Output: ['__add__', '__class__', '__contains__', '__delattr__', '__delitem__',
# '__delslice__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__',
# '__getitem__', '__getslice__', '__gt__', '__hash__', '__iadd__', '__imul__',
# '__init__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__',
# '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__',
# '__setattr__', '__setitem__', '__setslice__', '__sizeof__', '__str__',
# '__subclasshook__', 'append', 'count', 'extend', 'index', 'insert', 'pop',
# 'remove', 'reverse', 'sort']

import inspect

print(inspect.getmembers(str))
# Output: [('__add__', <slot wrapper '__add__' of ... ...

#####################################

mcase = {'a': 10, 'b': 34, 'A': 7, 'Z': 3}

mcase_frequency = {
    k.lower(): mcase.get(k.lower(), 0) + mcase.get(k.upper(), 0)
    for k in mcase.keys()
}

# mcase_frequency == {'a': 17, 'z': 3, 'b': 34}
#######

squared = {x ** 2 for x in [1, 1, 2]}
print(squared)
# Output: {1, 4}
################
multiples_gen = (i for i in range(30) if i % 3 == 0)
print(multiples_gen)
# Output: <generator object <genexpr> at 0x7fdaa8e407d8>
for x in multiples_gen:
    print(x)
    # Outputs numbers

#######################################################

try:
    file = open('test.txt', 'rb')
except IOError as e:
    print('An IOError occurred. {}'.format(e.args[-1]))
finally:
    print("This would be printed whether or not an exception occurred!")

# Output: An IOError occurred. No such file or directory
# This would be printed whether or not an exception occurred!

#########################


try:
    print('I am sure no exception is going to occur!')
except Exception:
    print('exception')
else:
    # any code that should only run if no exception occurs in the try,
    # but for which exceptions should NOT be caught
    print('This would only run if no exception occurs. And an error here '
          'would NOT be caught.')
finally:
    print('This would be printed in every case.')

# Output: I am sure no exception is going to occur!
# This would only run if no exception occurs. And an error here would NOT be caught
# This would be printed in every case.

###########################################
add = lambda x, y: x + y

print(add(3, 5))
# Output: 8

############################################ one-line
from pprint import pprint

my_dict = {'name': 'Yasoob', 'age': 'undefined', 'personality': 'awesome'}
print(dir(my_dict))
# ['__add__', '__class__', '__contains__', '__delattr__', '__delitem__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__imul__', '__init__', '__init_subclass__', '__iter__', '__le__', '__len__', '__lt__', '__mul__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rmul__', '__setattr__', '__setitem__', '__sizeof__', '__str__', '__subclasshook__', 'append', 'clear', 'copy', 'count', 'extend', 'index', 'insert', 'pop', 'remove', 'reverse', 'sort']

pprint(dir(my_dict))
# ['__add__',
#  '__class__',
#  '__contains__',
#  '__delattr__',
#  '__delitem__',

####

cat
file.json | python - m
json.tool
#################################

python - m
cProfile
my_script.py

##############################

python - c
"import csv,json;print json.dumps(list(csv.reader(open('csv_file.csv'))))"

############################################

a_list = [[1, 2], [3, 4], [5, 6]]
print(list(itertools.chain.from_iterable(a_list)))
# Output: [1, 2, 3, 4, 5, 6]

# or
print(list(itertools.chain(*a_list)))


# Output: [1, 2, 3, 4, 5, 6]

#######################################################
# One-Line Constructors

class A(object):
    def __init__(self, a, b, c, d, e, f):
        self.__dict__.update({k: v for k, v in locals().items() if k != 'self'})


###################################################################

for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print(n, 'equals', x, '*', n / x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')

#################################################################

import io

with open('photo.jpg', 'rb') as inf:
    jpgdata = inf.read()

if jpgdata.startswith(b'\xff\xd8'):
    text = u'This is a JPEG file (%d bytes long)\n'
else:
    text = u'This is a random file (%d bytes long)\n'

with io.open('summary.txt', 'w', encoding='utf-8') as outf:
    outf.write(text % len(jpgdata))
