from functools import wraps

# Our custom decorator that accepts an argument
def wrap(greeting=None):
    def wrapper_outer(fn):  # fn is the function being decorated
        @wraps(fn)          # preserves metadata of fn
        def wrapper_inner(*args, **kwargs):  # the actual function that runs
            # Print the greeting message and the first argument passed to the function
            print(f"{greeting} 👋 from wrapper and {args[0]}!")
            return fn(*args, **kwargs)       # call the original function (train)
        return wrapper_inner
    return wrapper_outer


@wrap(greeting="Hi")
def train(name):
    """This function greets someone by name."""
    # Print a message to greet the person by name
    print(f"Nice to meet you, {name}!")

train("Alice")