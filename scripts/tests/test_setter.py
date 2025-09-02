class A:
    def __init__(self):
        self._a = None

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        self._a = value
        print("i am in setter")


ins = A()
ins.a = 1
print(ins.a)
