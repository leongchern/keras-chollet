def addx(x, y):
    try:
        assert x > 5
    except AssertionError:
        print('x is below 5')
    return x + y

print(addx(1, 10))