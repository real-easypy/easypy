class SelectiveQueue():
    def __init__(self, initial_values = []):
        self.data = list(initial_values)

    def enqueue(self, value):
        self.data.append(value)

    def _pop(self, index):
        return self.data.pop(index)

    def dequeue(self, predicate = lambda _: True):
        for index in range(len(self.data)):
            if predicate(self.data[index]):
                return self._pop(index)
        return None

    def peek(self, predicate = lambda _: True):
        for value in self.data:
            if predicate(value):
                return value
        return None

    def move_values_to_end(self, *values):
        for value in values:
            self.dequeue(lambda x: value)

    def __str__(self):
        return str(self.data)

class SelectiveCycle(SelectiveQueue):
    def _pop(self, index):
        value = self.data.pop(index)
        self.data.append(value)
        return value
