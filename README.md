# Manhattan-waterworld

This is a repository for generating random Manhattan-World-style experiments
with ambiguous data association, multiple robots, and range measurements between
both robots and static beacons placed in the environment.

## Getting Started

To run some sample code:

``` Bash
# setup the environment (name assigned from the .yml file)
conda create --name manhattan_sim python=3.9 numpy matplotlib
conda activate manhattan_sim

# run the example code from the example directory of this repository
cd examples
python single_robot_random_trajectory.py
```

## Contributing

This repo is set up so that contributions will have consistent style and pass
some static type checks. The code styling is enforced through `black`, and helps
to improve the cleanliness, consistency, and readability of commits.

The static type checking is performed by `mypy` and helps to prevent bugs by
ensuring that object type expectations are being maintained. In addition, it
requires thoughtful type-annotation to improve the readability and
maintainability of the code in the future.

We run these through both `Github Actions` and the `pre-commit` framework, which
allow us to check our code before committing and then perform some quality tests
on the code once it has been committed. `Github Actions` does not require any
further setup to be used, the tests will be run when the code is pushed.

The [`pre-commit`](https://pre-commit.com/#intro) framework lets you make sure
the tests will pass before pushing the code. To set it up and use it all you
need to do is run the following commands:

``` Bash
# install pre-commit
pip3 install pre-commit
pre-commit install
```

Now, when you try to make a commit, pre-commit will run a series of tests that
must be passed. In the event that the code can be easily changed, mostly in the
case of reformatting, `pre-commit` will often make the change for you. From here
you can just run `git diff` to see the changes made, verify they're correct, add
the changes, and attempt the recommit.
