# Manhattan Simulator

This is a repository for generating random Manhattan-World-style 2D experiments.

While custom measurements are easy to add, currently supported measurement types include:

- robot odometry measurements
- pose-pose and pose-landmark loop closures (measurements are SE(2) measurements)
- pose-pose and pose-landmark range measurements
- ambiguous loop closures and ambiguous range measurements (the measurements above, but with uncertainty on the data association)

## Getting Started

Please look in our `example/` directory to see how to use this module to generate experiments.

Our Manhattan Simulator depends on our [PyFactorGraph](https://github.com/MarineRoboticsGroup/PyFactorGraph/) module, so
you will need to install PyFactorGraph via:

```bash
git clone git@github.com:MarineRoboticsGroup/PyFactorGraph.git
cd PyFactorGraph
pip3 install .
```

## Contributing

If you want to contribute a new feature to this package please read this brief section.

### Code Standards

Any necessary coding standards are enforced through `pre-commit`. This will run
a series of `hooks` when attempting to commit code to this repo. Additionally,
we run a `pre-commit` hook to auto-generate the documentation of this library to
make sure it is always up to date.

To set up `pre-commit`

```bash
cd ~/manhattan-world-sim
pip3 install pre-commit
pre-commit install
```

### Testing

If you want to develop this package and test it from an external package you can
also install via

```bash
cd ~/manhattan-world-sim
pip3 install -e .
```

The `-e` flag will make sure that any changes you make here are automatically
translated to the external library you are working in.
