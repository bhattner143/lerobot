import os
import multiprocessing

print(f"Total CPU cores (logical): {os.cpu_count()}")
print(f"Physical CPU cores: {multiprocessing.cpu_count()}")