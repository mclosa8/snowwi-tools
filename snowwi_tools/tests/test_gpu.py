#!/usr/bin/env python3

import pycuda.autoinit as p

print(p.device.name())
print(p.device.total_memory())
