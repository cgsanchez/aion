# Shared scientific toolchains

This project uses `/home/cgs/01_TOOLS/EasyBuild`.

- Read `easybuild-project.toml` before selecting native or Python tools.
- Follow `/home/cgs/01_TOOLS/EasyBuild/docs/AGENT_USAGE.md` for module loading,
  Conda isolation, provenance, and dependency requests.
- Use already-published compatible modules freely and record them before
  claiming reproducibility.
- Do not mutate shared infrastructure or improvise a missing native dependency
  through Conda, copied libraries, untracked binaries, or system packages.
- The project owns its source layout, build system, build directories, flags,
  and test workflow.
