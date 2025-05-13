from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class SimDataCustomV0(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ^ F: This is the basic embedding we are using (i would start with this, then maybe it's a direction to explore)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        # TODO: add proper documentation to each field (specifying meaning, utils and shapes)
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        # ^ F: THIS IS MAIN OBSERVATION - MANDATORY
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        # ? F: DO WE NEED THIS? WE MIGHT ACTUALLY PROVIDE MORE INFORMATION AT TRAINING TIME
                        # ? AND THEN AT TEST TIME ONLY USE IMAGE + PROMPT
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        # ^ F: THIS IS THE STATE OF THE ROBOT ARM - MANDATORY
                        # ? Maybe need to change the shape
                        'state': tfds.features.Tensor(
                            shape=(8,), # TODO penso sia 8 corretto perché ci va il padding -> vedi config
                            dtype=np.float32,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position].',
                        ),
                        'object_pose': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Object pose, consists of [x, y, z, w, x, y, z].', # pos + quat in isaacsim notation
                        ),
                        'target_pose': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Target pose, consists of [x, y, z, w, x, y, z].', # pos + quat in isaacsim notation
                        ),
                        'camera_pose': tfds.features.Tensor(
                            shape=(6,),
                            dtype=np.float32,
                            doc='Camera pose, consists of [x, y, z, target_x, target_y, target_z].', # pos + quat in isaacsim notation
                        ),
                    }),
                    # ^ F: THIS IS THE ACTION THE ROBOT IS TAKING - MANDATORY (we are not using the last scalar)
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    # ! F: Discount and Reward are not used in out implementation, but we CANNOT remove them
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    # ! F: This part is not explicitly used in our implementation, but we CANNOT remove it
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    # ^ F: THIS IS THE LANGUAGE INSTRUCTION - MANDATORY
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                # ! F: This part is something we have to come up with, since we CANNOT remove it
                # ! Can simply be the name of the file
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([step['language_instruction']])[0].numpy()

                episode.append({
                    'observation': {
                        'image': step['image'],
                        'wrist_image': step['wrist_image'],
                        'state': step['state'],
                        'object_pose': step['object_pose'][0],
                        'target_pose': step['target_pose'][0],
                        'camera_pose': step['camera_pose'][0],
                    },
                    'action': step['action'],
                    'discount': 1.0,
                    'reward': float(i == (len(data) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data) - 1),
                    'is_terminal': i == (len(data) - 1),
                    'language_instruction': step['language_instruction'],
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # ^ Check which folders is the script taking data from

        # create list of all examples
        episode_paths = glob.glob(path)
        print("I am here", episode_paths)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

