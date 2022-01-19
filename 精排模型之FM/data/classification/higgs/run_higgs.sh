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

# Training task:
#  -s : 0    (use LR for classification)
#  -x : acc  (use accuracy metric)
# The model will be stored in higgs-train.csv.model
../../xlearn_train ./higgs-train.csv -s 0 -v ./higgs-test.csv -x acc
# Prediction task:
# The output result will be stored in higgs-test.csv.out
../../xlearn_predict ./higgs-test.csv ./higgs-train.csv.model