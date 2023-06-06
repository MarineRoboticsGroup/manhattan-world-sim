# Manhattan Simulator

This is a repository for generating random Manhattan-World-style 2D experiments. 

While custom measurements are easy to add, currently supported measurement types include:

- robot odometry measurements
- pose-pose and pose-landmark loop closures (measurements are SE(2) measurements)
- pose-pose and pose-landmark range measurements
- ambiguous loop closures and ambiguous range measurements (the measurements above, but with uncertainty on the data association)

## Getting Started

Please look in our `example/` directory to see how to use this module to generate experiments.

Our Manhattan Simulator depends on our [https://github.com/MarineRoboticsGroup/PyFactorGraph/](PyFactorGraph) module, so
you will need to install PyFactorGraph via:

```bash
git clone git@github.com:MarineRoboticsGroup/PyFactorGraph.git
cd PyFactorGraph
pip install .
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
pre-commit install # must run this command in the root of this repo
```

Now, when you try to make a commit, pre-commit will run a series of tests that
must be passed. In the event that the code can be easily changed, mostly in the
case of reformatting, `pre-commit` will often make the change for you. From here
you can just run `git diff` to see the changes made, verify they're correct, add
the changes, and attempt the recommit.
