# R&D RL-demo

This is the main repository for Rice Robotics Club R&D's reinforcement learning environments for the quadruped robot project.

## Setting up the Environment

The environment setup currently uses Conda. This is a more robust system on top of Python and Pip that better supports
multiple platforms, especially with PyBullet on Mac computers.

1. First, install miniconda here to be able to load the environment:
https://www.anaconda.com/download/success

2. Then, once conda is installed, you can load the environment from the `environment.yml` file in the root of this repo:
    ```shell
    conda env create -f environment.yml
    ```

3. This will create a conda environment called `rl-demo`. You can activate the environment using the following command:
    ```shell
    conda activate rl-demo
    ```

4. You can verify that the environment was loaded by the text `(rl-demo)` appearing to the left of your shell prompt.
From here, you can then run the python files in this repository and all the necessary packages will be present in the
environment:
    ```shell
    python train.py
    ```

5. To update the environment, in the case that `environment.yml` changes, you can run this command:
    ```shell
    conda env update --file environment.yml --prune
    ```
