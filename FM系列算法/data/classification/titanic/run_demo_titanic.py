# Copyright (c) 2018 by contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import xlearn as xl

# Training task
fm_model = xl.create_fm()  # Use factorization machine
fm_model.setTrain("./titanic_train.txt")  # Training data

# param:
#  0. Binary classification task
#  1. learning rate: 0.2
#  2. lambda: 0.002
#  3. metric: accuracy
param = {'task':'binary', 'lr':0.2, 
         'lambda':0.002, 'metric':'acc'}

# Use cross-validation
fm_model.cv(param)