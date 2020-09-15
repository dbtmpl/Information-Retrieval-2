# Abusing the structure just to test the QL model
from modeling.qulac_ql import test_model

# Add all configs and startup code here. Maybe a good idea if:
# 1. Parse configs and other setup code (probably what OpenNIR wants)
# 2. Init Trainer / OpenNIR pipeline with configs
# 3. From there everything should be encapsulated in the Trainer OpenNIR pipeline

test_model()
