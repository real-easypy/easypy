from collections import defaultdict
from contextlib import ExitStack


class KeyedStack(ExitStack):
    def __init__(self, context_factory):
        self.context_factory = context_factory
        self.contexts_dict = defaultdict(list)
        super().__init__()

    def enter_context(self, *key):
        cm = self.context_factory(*key)
        self.contexts_dict[key].append(cm)
        super().enter_context(cm)

    def exit_context(self, *key):
        self.contexts_dict[key].pop(-1).__exit__(None, None, None)

