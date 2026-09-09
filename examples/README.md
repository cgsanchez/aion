# Aion 0.2 examples

These are deliberately small interface examples, not production calculations.
Each writes beneath `example-output/` by default and refuses to overwrite an
existing immutable artifact.

Run them from the repository root in the managed environment:

```bash
/home/cgs/01_TOOLS/EasyBuild/conda/envs/aion/bin/python \
  examples/01_h2_reference_casida.py

/home/cgs/01_TOOLS/EasyBuild/conda/envs/aion/bin/python \
  examples/02_lih_four_formulations.py

/home/cgs/01_TOOLS/EasyBuild/conda/envs/aion/bin/python \
  examples/03_checkpoint_resume_cli.py
```

Every script accepts `--output DIRECTORY` and `--backend cpu|gpu`. For GPU
execution, launch the script through `tools/gpu-python`, for example:

```bash
tools/gpu-python examples/02_lih_four_formulations.py --backend gpu
```

The examples cover exactly three workflows:

1. prepare an H2 reference and compute a structured Casida spectrum;
2. propagate LiH with bare LG, bare VG, and P0+E1 in its length and velocity
   representations using one pulse definition;
3. generate normalized TOML, execute the CLI, and resume a checkpoint taken
   immediately before a kick boundary.

The short pulse and minimal bases keep the examples quick. Their numerical
parameters are pedagogical and must not be reused as scientific convergence
choices.
