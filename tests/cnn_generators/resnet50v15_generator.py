#!/usr/bin/env python3

import numpy as np
import pydtnn

class Params:
    pass

p = Params()
p.model_name = "resnet50v15_imagenet"
p.dataset = "imagenet"
p.use_synthetic_data = True
p.tensor_format = "NHWC"
p.dtype = np.float32
p.batch_size = 1
p.steps_per_epoch = 10

m = pydtnn.Model(**vars(p))
m.print_in_convdirect_format()
