# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Dummy reward function to disable default reward computation.

When using custom reward computation in ray_trainer that requires old_log_probs,
use this dummy function to bypass the default reward computation which happens
before old_log_probs are available.
"""


async def dummy_reward(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """
    Dummy reward function that returns 0.0.
    
    This is used as a placeholder when custom reward computation is enabled
    in the trainer, which has access to old_log_probs and other computed values.
    
    Args:
        data_source: The source identifier for the data
        solution_str: The generated response/solution from the model
        ground_truth: The ground truth answer
        extra_info: Additional metadata
        **kwargs: Any custom parameters passed via reward_kwargs
    
    Returns:
        float: Always returns 0.0 since actual reward is computed later
    """
    return 0.0
