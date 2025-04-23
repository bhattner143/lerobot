#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from typing import Iterator, Union

import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        episode_data_index: dict,
        episode_indices_to_use: Union[list, None] = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """
        Sampler that optionally incorporates episode boundary information.

        Args:
            episode_data_index (dict): Dictionary containing episode boundary information.
            It must have two keys:
            - 'from': A list of start indices for each episode.
            - 'to': A list of end indices for each episode.
            episode_indices_to_use (list or None): A list of episode indices to include in the sampling.
            If None, all episodes are included. Assumes episodes are indexed from 0 to N-1.
            drop_n_first_frames (int): Number of frames to drop from the start of each episode.
            This is useful for ignoring initial frames that might be irrelevant.
            drop_n_last_frames (int): Number of frames to drop from the end of each episode.
            This is useful for ignoring trailing frames that might be irrelevant.
            shuffle (bool): Whether to shuffle the indices during iteration. If True, indices are shuffled.

        Attributes:
            indices (list): A list of indices that will be sampled, after applying the episode filtering
            and frame-dropping logic.
            shuffle (bool): Whether to shuffle the indices during iteration.
        """
        indices = []
        # Iterate over each episode's start and end indices
        for episode_idx, (start_index, end_index) in enumerate(
            zip(episode_data_index["from"], episode_data_index["to"], strict=True)
        ):
            # Check if the current episode should be included
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                # Add the range of indices for the episode, excluding the dropped frames
                indices.extend(
                    range(start_index.item() + drop_n_first_frames, end_index.item() - drop_n_last_frames)
                )

        # Store the computed indices
        self.indices = indices
        # Store the shuffle flag
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        """
        Iterator method to yield indices.

        If shuffle is enabled, the indices are shuffled before yielding.
        Otherwise, the indices are yielded in order.

        Yields:
            int: The next index in the sampling sequence.
        """
        if self.shuffle:
            # Shuffle the indices using torch.randperm and yield them
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            # Yield the indices in order
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        """
        Returns the total number of indices available for sampling.

        Returns:
            int: The number of indices.
        """
        return len(self.indices)
